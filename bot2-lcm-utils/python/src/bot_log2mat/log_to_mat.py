    #!/usr/bin/python
#
#Converts a LCM log to a "matrix" format that is easier to work with in external
#tools such as Matlab. The set of messages on a given channel can be represented 
#as a matrix, where the columns of this matrix are the the fields of the lcm type
#with one message per row

import os
import sys
import binascii
import types
import numpy
import re
import getopt

# check which version for mio location
if sys.version_info < (2, 6):
    import scipy.io.mio
else:
    import scipy.io.matlab.mio

from scipy.io import savemat

from lcm import EventLog
from scan_for_lcmtypes import *

def usage():
    pname, sname = os.path.split(sys.argv[0])
    sys.stderr.write("usage: % s %s < filename > \n" % (sname, str(longOpts)))
    print """
    -h --help                 print this message
    -p --print                Output log data to stdout instead of to .mat
    -f --format               print the data format to stderr
    -s --seperator=sep        print data with separator [sep] instead of default to ["" ""]
    -c --channelsToProcess=chan        Parse channelsToProcess that match Python regex [chan] defaults to [".*"]
    -i --ignore=chan          Ignore channelsToProcess that match Python regex [chan]
                              ignores take precedence over includes!
    -o --outfile=ofname       output data to [ofname] instead of default [filename.mat or stdout]
    -l --lcmtype_pkgs=pkgs    load python modules from comma seperated list of packages [pkgs] defaults to ["botlcm"]
    -v                        Verbose

    """
    sys.exit()

flatteners = {}
data = {}
rawdata = {}

def make_simple_accessor(fieldname):
    return lambda lst, x: lst.append(getattr(x, fieldname))

def make_numpy_array_accessor(fieldname):
    return lambda lst, x: lst.extend(numpy.array(getattr(x, fieldname)).ravel())

def make_obj_accessor(fieldname, func):
    return lambda lst, x: func(lst, getattr(x, fieldname))

def make_obj_list_accessor(fieldname, func):
    return lambda lst, x: map(lambda item: func(lst, item), getattr(x, fieldname))
#    def list_accessor(lst, msg):
#        msg_lst = getattr(msg, fieldname)
#        for elem in msg_lst:
#            func(lst, elem)
#    return list_accessor
#

# def get_base_array(val):
#     if type(val) in [ types.IntType, types.LongType, types.FloatType,
#                       types.BooleanType ]:
#         return numpy.array([], dtype=double)
#     elif type(val) in [ types.ListType, types.TupleType ]:
#         return basestruct[fieldname] = numpy.array([], dtype=double)


def make_lcmtype_accessor(msg):
    funcs = []

    for fieldname in getattr(msg, '__slots__'):
        m = getattr(msg, fieldname)

        if type(m) in [ types.IntType, types.LongType, types.FloatType,
                types.BooleanType ]:
            # scalar
            accessor = make_simple_accessor(fieldname)
            funcs.append(accessor)
        elif type(m) in [ types.ListType, types.TupleType ]:
            # convert to a numpy array
            arr = numpy.array(m)

            # check the data type of the array
            if arr.dtype.kind in "bif":
                # numeric data type
                funcs.append(make_numpy_array_accessor(fieldname))
            elif arr.dtype.kind == "O":
                # compound data type
                typeAccess = make_lcmtype_accessor(m[0])
                funcs.append(make_obj_list_accessor(fieldname, typeAccess))
                #pass
        elif type(m) in types.StringTypes:
            # ignore strings
            pass
        else:
            funcs.append(make_obj_accessor(fieldname, make_lcmtype_accessor(m)))

    def flatten(lst, m):
        for func in funcs:
            func(lst, m)
    return flatten

def make_flattener(msg):
    accessor = make_lcmtype_accessor(msg)
    def flattener(m):
        result = []
        accessor(result, m)
        return result
    return flattener




def make_lcmtype_string(msg, base=True):
    typeStr = []
    count = 0
    for fieldname in getattr(msg, '__slots__'):
        m = getattr(msg, fieldname)

        if type(m) in [ types.IntType, types.LongType, types.FloatType, types.BooleanType ]:
            count = count + 1
            if base:
                typeStr.append("%d- %s" % (count, fieldname))
            else:
                typeStr.append(fieldname)
        elif type(m) in [ types.ListType, types.TupleType ]:
            # convert to a numpy array
            arr = numpy.array(m)
            # check the data type of the array
            if arr.dtype.kind in "bif":
                # numeric data type

                if base:
                    typeStr.append("%d- %s(%d)" % (count + 1, fieldname, len(arr.ravel())))
                else:
                    typeStr.append("%s(%d)" % (fieldname, len(arr.ravel())))
                count = count + len(arr.ravel())
            elif arr.dtype.kind == "O":
                # compound data type
                subStr, subCount = make_lcmtype_string(m[0], False)
                numSub = len(m)
                if base:
                    subStr = "%d- %s<%s>(%d)" % (count + 1, fieldname, ", ".join(subStr), numSub)
                else:
                    subStr = "%s<%s>(%d)" % (fieldname, ", ".join(subStr), numSub)
                typeStr.append(subStr)
                count = count + numSub * subCount
                #pass
        elif type(m) in types.StringTypes:
            # ignore strings
            pass
        else:
            subStr, subCount = make_lcmtype_string(m, False);
            if base:
                for s in subStr:
                    typeStr.append("%d- %s.%s" % (count+1, fieldname , s))
                    count = count + subCount
            else:
                count = count + subCount
                for s in subStr:
                    typeStr.append(fieldname + "." + s)

    return typeStr, count

def deleteStatusMsg(statMsg):
    if statMsg:
        sys.stderr.write("\r")
        sys.stderr.write(" " * (len(statMsg)))
        sys.stderr.write("\r")
    return ""

# Get the data type of the lowest level, if everything is empty returns list type
def getUnderlyingType(val):
    if type(val) in [ types.ListType, types.TupleType ]:
        # Recursively dive for the right type
        for listval in val:
            newType = getUnderlyingType(listval)
            if (newType != types.ListType):
                return newType

        # Nothing had a valid type, so return list
        return types.NoneType

    # Wasn't a list/tuple, so just return
    return type(val)

def convertSingleDict(origDict):
    for field in origDict:
        # Check that this is actually a list
        if type(origDict[field]) in [ types.ListType, types.TupleType ]:
            # If use, find the underlying type
            ftype = getUnderlyingType(origDict[field])
            if (ftype == types.NoneType):
                continue;
            elif ftype in types.StringTypes:
                origDict[field] = numpy.array(origDict[field], dtype=object).transpose()
            elif ftype in [ types.IntType, types.LongType, types.BooleanType, types.FloatType ]:
                # Find the next type
                if (type(origDict[field][0]) in [ types.IntType, types.LongType, types.BooleanType, types.FloatType ]):
                    # Single type class
                    origDict[field] = numpy.atleast_2d(numpy.array(origDict[field], dtype=float))
                else:
                    origData = origDict[field]
                    # Convert underlying elements to float
                    newData = []
                    for elem in origData:
                        newData.append(numpy.array(elem, dtype=float))

                    newData = numpy.array(newData).transpose()

                    origDict[field] = newData

        elif type(origDict[field]) in [ types.IntType, types.LongType, types.BooleanType ]:
            origDict[field] = numpy.array(origDict[field], dtype=float)

    return origDict


# Take in a dict, squash numeric data of the same size, convert numeric data to double and char data to string
def makeArrayDict(dictIn):
    dictOut = {}
    for channel in dictIn:
        dictOut[channel + 'Parsed'] = convertSingleDict(dictIn[channel])

    return dictOut

def addMessageList(field):
    # Check that his function wasn't called incorrectly
    basetype = getUnderlyingType(field)
    if not hasattr(basetype, '__slots__'):
        raise ValueError('Tried to run addMessageList with a non-lcm base type')

    # Check if we are only one level away
    if (len(field) == 0):
        return {}

    if hasattr(field[0], '__slots__'):
        # Only one level down, first create the empty structure
        # Outputs a structure of concatenated data, concatenated along the list
        msgStruct = addMessage({}, '', field[0], True)
        msgStruct['numMsg'] = 0
        # Add everything to the structure
        for item in field:
            msgStruct = addMessage(msgStruct, '', item)
            msgStruct['numMsg'] += 1

        # Convert the dictionary to approriate form
        return convertSingleDict(msgStruct)

    # At least one level of lists lie between this and the underlying messeges
    # This will output a list of struct/lists (depending on the next level
    outList = []
    for item in field:
        outList.append(addMessageList(item))

    return outList

def makeVarName(baseName, fieldName):
    if len(baseName) == 0:
        return fieldName
    else:
        return baseName + '__' + fieldName

def addMessage(struct, baseName, msg, create = False):
    for fieldName in getattr(msg, '__slots__'):
        field = getattr(msg, fieldName)
        if hasattr(field, '__slots__'):
            # The field is another lcm type
            struct = addMessage(struct, makeVarName(baseName, fieldName), field, create)
        else:
            basetype = getUnderlyingType(field)
            if hasattr(basetype, '__slots__'):
                # The field is a tuple/list of another lcmtype
                if (create):
                    struct[makeVarName(baseName, fieldName)] = []
                else:
                    # Appends either a struct, or a list of structs, or...
                    struct[makeVarName(baseName, fieldName)].append(addMessageList(field))

            else:
                # Non message field, we can just append
                if (create):
                    struct[makeVarName(baseName, fieldName)] = []
                else:
                    struct[makeVarName(baseName, fieldName)].append(field)

    return struct

longOpts = ["help", "print", "format", "separator", "channelsToProcess", "ignore", "outfile", "lcm_packages"]

### Start of processing
try:
    opts, args = getopt.gnu_getopt(sys.argv[1:], "hpvfs:c:i:o:l:", longOpts)
except getopt.GetoptError, err:
    # print help information and exit:
    print str(err) # will print something like "option -a not recognized"
    usage()
if len(args) != 1:
    usage()
#default options
fname = args[0]
lcm_packages = [ "botlcm"]

outDir, outFname = os.path.split(os.path.abspath(fname))
outFname = outFname.replace(".", "_")
outFname = outFname.replace("-", "_")
outFnameRaw = outDir + "/" + outFname + "_parsed.mat"
outFname = outDir + "/" + outFname + ".mat"
printFname = "stdout"
printFile = sys.stdout
verbose = False
printOutput = False
printFormat = False
channelsToIgnore = ""
checkIgnore = False
channelsToProcess = ".*"
separator = ' '
for o, a in opts:
    if o == "-v":
        verbose = True
    elif o in ("-h", "--help"):
        usage()
    elif o in ("-p", "--print"):
        printOutput = True
    elif o in ("-f", "--format"):
        printFormat = True
    elif o in ("-s", "--separator="):
        separator = a
    elif o in ("-o", "--outfile="):
        outFname = a
        printFname = a
    elif o in ("-c", "--channelsToProcess="):
        channelsToProcess = a
    elif o in ("-i", "--ignore="):
        channelsToIgnore = a
        checkIgnore = True
    elif o in ("-l", "--lcm_packages="):
        lcm_packages = a.split(",")
    else:
        assert False, "unhandled option"

fullPathName = os.path.abspath(outFname)
dirname = os.path.dirname(fullPathName)
outBaseName = ".".join(os.path.basename(outFname).split(".")[0:-1])
fullBaseName = dirname + "/" + outBaseName

type_db = make_lcmtype_dictionary()

channelsToProcess = re.compile(channelsToProcess)
channelsToIgnore = re.compile(channelsToIgnore)
log = EventLog(fname, "r")

if printOutput:
    sys.stderr.write("opened % s, printing output to %s \n" % (fname, printFname))
    if printFname == "stdout":
        printFile = sys.stdout
    else:
        printFile = open(printFname, "w")
else:
    sys.stderr.write("opened % s, outputing to % s\n" % (fname, outFname))

ignored_channels = []
msgCount = 0
statusMsg = ""
startTime = 0

### Start processing the log ###
for e in log:
    if msgCount == 0:
        startTime = e.timestamp

    if e.channel in ignored_channels:
        continue
    if ((checkIgnore and channelsToIgnore.match(e.channel) and len(channelsToIgnore.match(e.channel).group())==len(e.channel)) \
         or (not channelsToProcess.match(e.channel))):
        if verbose:
            statusMsg = deleteStatusMsg(statusMsg)
            sys.stderr.write("ignoring channel %s\n" % e.channel)
        ignored_channels.append(e.channel)
        continue

    ## This is an event we actually want to process
    packed_fingerprint = e.data[:8]
    lcmtype = type_db.get(packed_fingerprint, None)
    if not lcmtype:
        if verbose:
            statusMsg = deleteStatusMsg(statusMsg)
            sys.stderr.write("ignoring channel %s -not a known LCM type\n" % e.channel)
        ignored_channels.append(e.channel)
        continue
    try:
        msg = lcmtype.decode(e.data)
    except:
        statusMsg = deleteStatusMsg(statusMsg)
        sys.stderr.write("error: couldn't decode msg on channel %s\n" % e.channel)
        continue

    ## We were successfully able to decode the message
    msgCount = msgCount + 1
    if (msgCount % 5000) == 0:
        statusMsg = deleteStatusMsg(statusMsg)
        statusMsg = "read % d messages, % d %% done" % (msgCount, log.tell() / float(log.size())*100)
        sys.stderr.write(statusMsg)
        sys.stderr.flush()

    ## Figure out how to parse this message
    if e.channel in flatteners:
        flattener = flatteners[e.channel]
    else:
        flattener = make_flattener(msg)
        flatteners[e.channel] = flattener
        data[e.channel] = []
        # Create empty data structure with expanded field
        basestruct = addMessage({}, '', msg, True)

        basestruct['logTime'] = []
        basestruct['channel'] = str(e.channel)
        basestruct['typename'] = msg.__class__.__name__
        basestruct['numMsg'] = 0

        rawdata[e.channel] = basestruct

        if printFormat:
            statusMsg = deleteStatusMsg(statusMsg)
            typeStr, fieldCount = make_lcmtype_string(msg)
            typeStr.append("%d- log_timestamp" % (fieldCount + 1))

            typeStr = "\n#%s  %s :\n#[\n#%s\n#]\n" % (e.channel, lcmtype, "\n#".join(typeStr))
            sys.stderr.write(typeStr)


    ## Parse the message
    a = flattener(msg)
    #in case the initial flattener didn't work for whatever reason :-/
    # convert to a numpy array
    arr = numpy.array(a)
    # check the data type of the array
    if not(arr.dtype.kind in "bif"):
        statusMsg = deleteStatusMsg(statusMsg)
        sys.stderr.write("WARNING: needed to create new flattener for channel %s\n" % (e.channel))
        flattener = make_flattener(msg)
        flatteners[e.channel] = flattener
        a = flattener(msg)

    ## Compute the log time
    logTime = (e.timestamp - startTime) / 1e6
    a.append(logTime)

    ## Place the new data in a structure
    if printOutput:
        printFile.write("%s%s%s\n" % (e.channel, separator, separator.join([str(k) for k in a])))
    else:
        data[e.channel].append(a)
        rawdata[e.channel] = addMessage(rawdata[e.channel], '', msg)
        rawdata[e.channel]['logTime'].append(logTime)
        rawdata[e.channel]['numMsg'] += 1


deleteStatusMsg(statusMsg)
if not printOutput:
    #need to pad variable length messages with zeros...
    for chan in data:
        lengths = map(len, data[chan])
        maxLen = max(lengths)
        minLen = min(lengths)
        if maxLen != minLen:
            sys.stderr.write("padding channel %s with zeros, messages ranged from %d to %d \n" % (chan, minLen, maxLen))
            count = 0
            for i in data[chan]:
                pad = numpy.zeros(maxLen - lengths[count])
                i.extend(pad)
                count = count + 1


    sys.stderr.write("loaded all %d messages, saving to % s\n" % (msgCount, outFname))

    if sys.version_info < (2, 6):
        scipy.io.mio.savemat(outFname, data)
    else:
        scipy.io.matlab.mio.savemat(outFname, data, oned_as='row')

    d = makeArrayDict(rawdata)

    savemat(outFnameRaw, d, oned_as='column')

    ## Write the actual file
    mfile = open(dirname + "/" + outBaseName + ".m", "w")
    ## Write the .m to load it
    loadFunc = """function [d imFnames]=%s()
full_fname = '%s';
fname = '%s';
if (exist(full_fname,'file'))
    filename = full_fname;
else
    filename = fname;
end
d = load(filename);
""" % (outBaseName, outFname, fullPathName)



    mfile.write(loadFunc);
    mfile.close()
