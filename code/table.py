# simple table abstraction
import numpy as np
import sys
import matplotlib.pyplot as plt
import uuid
import csv

def cvt(item) :
    """Attempt to convert string to value it represents"""
    item = item.strip()          # remove leading and trailing whitespace
    if not item : return ""
    if item[0] == "$": return float(item.strip("$,"))
    if item.strip("0123456789,.") : return item
    item = item.strip(",")
    try: return int(item)   # try to convert it to an integer
    except ValueError:
        return float(item) # otherwise a float



def getType(item) :
    """Attempt to determine what type of value a string represents"""
    item = item.strip()         
    if not item : return "NULL"
    if item[0] == "$": return "$"
    if item.strip("0123456789,.") : return "STR"
    return "NUM"

def getRanges(scol) :
    """Get sequence of ranges equal elements in a sorted array."""
    ranges = []             
    low = 0
    val = scol[low]
    for i in range(0,len(scol)) :
        if scol[i] != val : 
            ranges.append((low,i))
            low = i
            val = scol[i]
    ranges.append((low,len(scol)))
    return ranges

def getStarts(ranges) :
    """Get the start index for each range in a sequence."""
    return [s for (s,e) in ranges]

def getDistinct(a) :
    res = []
    for item in a : 
        if item not in res : res.append(item)
    if len(res)==1 : return res[0]
    else : return res

class table :
    """table class providing operations on named columns of data."""

    tables = []                 # list of current tables

    @staticmethod
    def list() :
        """List all tables in a text format."""
        for x in table.tables : 
            print "{0}[{1}x{2}] : {3}".format(x.name,len(x.cols),x.len, 
                                              "".join(str(item)+ ' : ' for item in x.cols))

    @staticmethod
    def get(name) :
        """Get a named table."""
        for x in table.tables :
            if x.name == name : return x
        return []

    @staticmethod
    def rename(oldName, newName) :
        """Get a named table."""
        for x in table.tables :
            if x.name == oldName : 
                x.name = newName
                return x

    @staticmethod
    def drop(name) :
        """Drop a named table."""
        for x in table.tables :
            if x.name == name : 
                tables.remove(x) # found the table, remove it from list
                return

    def __init__(self, name=[], cols=[]) :
        """Create a table.
        A table consists of an order sequence of named columns of homogeneous type
        .cols - sequence of column names
        .type - type of each column {num,str,dollar} - ot implemented
        .len - length of the columns
        .data - dictionary colName : columnData
        """
        if name :
            self.name = name    # give the table a name
        else :
            self.name = uuid.uuid1().hex # make one up
        self.cols = cols        # ordered sequence of column names
        self.len = 0            # zero rows
        self.type = ['unk' for c in cols]          # unknown type
        self.data = {c:np.array([]) for c in cols} # create empty columns
        self.tables.append(self)        # add to manifest

    def setColNames(self,cols,types) :
        """Initiatize a table with specified column names."""
        self.len = 0            # zero rows
        self.cols = cols        # attach the column names 
        self.type = types       # attach the types
        self.data = {c:np.array([]) for c in cols} # create empty columns
        return self

    def addRow(self, row) :
        """Add a row of data that is conformant with the columns."""
        assert len(row) == len(self.cols), "row has wrong number of elements"
        for item, colName in zip(row, self.cols) :
            self.data[colName] = np.append(self.data[colName], item) # extend each column
        self.len = self.len + 1                 # increment the number of rows
        return self

    def row(self,n) :
        """Return a dict consisting of the indexed row tagged y col name."""
        return {colName:self.data[colName][n] for colName in self.cols}

    def addCol(self, colName, colType, col) :
        """Add a named column.  If already exists, replace it. Return table."""
        if not self.cols : 
            self.cols = [colName] # first column name
            self.type = [colType] # associate type
            self.len = len(col) # set length to that of first column
        else : 
            assert len(col) == self.len, "Column length does not match table."
            if colName not in self.cols :
                self.cols.append(colName) # append new column
                self.type.append(colType) # associate type
            else :
                self.type[self.cols.index(colName)] = colType
        self.data[colName] = np.array(col)
        return self

    def col(self,colName) :
        """Return a sequence consisting of the named column."""
        return self.data[colName]

    def show(self, maxrows=30) :
        """Print the contents of a table as text."""
        print "".join(str(colName)+ '\t' for colName in self.cols)
        for r in range(0,min(self.len,maxrows)) :
            for colName,colType in zip(self.cols,self.type) :
                v = self.data[colName][r]
                if colType == "$" : print '$',
                print str(v)+'\t',
            print

    @staticmethod
    def csv(name, csvfilename) :
        """Create a table from a csv file, converting text to vals where possible."""
        with open(csvfilename, 'rU') as csvfile:
            rowreader = csv.reader(csvfile)
            header = rowreader.next()
            rows = []
            row = rowreader.next() # get first row
            types = map(getType, row) # guess the types
            rows.append(map(cvt,row)) # record the data
            for row in rowreader :    # do the rest of the rows
                rows.append(map(cvt,row))

            t = table(name)
            for c,colName in zip(range(0,len(header)), header) :
                col =  [r[c] for r in rows]
                t.addCol(colName, types[c], col)
            return t
                
    def plot(self, *args, **kwargs) :
        """Plot a collection of named columns."""
        for colName in args :
            s = self.col(colName)
            plt.plot(s, **kwargs)
#        if 'ylim' in kwargs.keys() : plt.ylim(kwargs['ylim'])
        plt.show(block=False)
        return s

    def scatter(self, xCol, yCol, **kwargs) :
        """Scatter plot of two named columns."""
        plt.scatter(self.col(xCol),self.col(yCol), **kwargs)
        return plt.show(block=False)

    def barh(self,labCol, valCol) :
        """Bar graph of a sequence consisting of the named column."""
        ypos = np.arange(self.len)
        plt.barh(ypos, self.col(valCol), alpha=0.8)
        plt.xlabel(valCol)
        plt.yticks(ypos, self.col(labCol), stretch='ultra-condensed')
        plt.show(block=False)

    def map(self, fun, rCol, rType, *args) :
        """Create new column by applying a function to set of existing ones."""
        cargs = [self.col(a) for a in args]
        res = apply(fun,cargs)
        return self.addCol(rCol, rType, res)

    def add(self, rCol, aCol, bCol) :
        """Result column is created as aCol + bCol."""
        assert self.type[self.cols.index(aCol)] == 'NUM' and self.type[self.cols.index(bCol)] == 'NUM',"Columns must be NUM"
        return self.addCol(rCol, 'NUM',  self.col(aCol) + self.col(bCol))
        
    def sub(self, rCol, aCol, bCol) :
        """Result column is created as aCol - bCol."""
        assert self.type[self.cols.index(aCol)] == 'NUM' and self.type[self.cols.index(bCol)] == 'NUM',"Columns must be NUM"
        return self.addCol(rCol, 'NUM', self.col(aCol) - self.col(bCol))

    def mul(self, rCol, aCol, bCol) :
        """Result column is created as aCol * bCol."""
        assert self.type[self.cols.index(aCol)] == 'NUM' and self.type[self.cols.index(bCol)] == 'NUM',"Columns must be NUM"
        return self.addCol(rCol, 'NUM', self.col(aCol) * self.col(bCol))

    def div(self, rCol, aCol, bCol) :
        """Result column is created as aCol / bCol."""
        assert self.type[self.cols.index(aCol)] == 'NUM' and self.type[self.cols.index(bCol)] == 'NUM',"Columns must be NUM"
        return self.addCol(rCol, 'NUM', self.col(aCol) / self.col(bCol))

    def scale(self, rCol, aCol, factor) :
        """Result column is created as factor * aCol."""
        assert self.type[self.cols.index(aCol)] == 'NUM', "Column must be NUM"
        return self.addCol(rCol, 'NUM', factor*self.col(aCol))

    def neg(self, rCol, aCol) :
        """Result column is created as -aCol."""
        assert self.type[self.cols.index(aCol)] == 'NUM', "Column must be NUM"
        return self.addCol(rCol, 'NUM', -self.col(aCol))

    def reduce(self, fun2, colName, initializer=None) :
        assert self.type[self.cols.index(aCol)] == 'NUM', "Column must be NUM"
        col = self.col(colName)
        if self.len <= 0 : return initializer
        if initializer is None : 
            if self.len == 1 : return col[0]
            r = col[0]
            col = col[1:]
        else : r = initializer
        for v in col : r = fun2(r,v)
        return r

    def index(self, colName) :
        self.addCol(colName,'NUM', range(0,self.len))

    def sort(self, colName, direction="increasing") :
        """Sort table by named column"""
        if direction == "increasing" :
            index = np.argsort(self.col(colName))
        else :
            index = np.argsort(-self.col(colName))        
        for key in self.cols :
            self.data[key]=self.data[key][index]
        
    def histogram(self,colName, plot=True, **kwargs) :
        counts,bins,patches = plt.hist(self.col(colName),**kwargs)
        if plot : plt.show()
        return {"counts":counts,"bins":bins}

    def stats(self,colName) :
        colType = self.type[self.cols.index(colName)]
        if colType != 'STR' :
            col = self.col(colName)
            n = float(self.len)
            cmax = np.max(col)
            cmin = np.min(col)
            csum = np.sum(col)
            cmean = csum/n
            cmedian = np.median(col)
            cvar = np.sum((col-cmean)*(col-cmean))/n
            cstd = np.sqrt(cvar)
            return {'count':n, 'min':cmin, 'max':cmax, 'sum':csum, 'mean':cmean, 'var':cvar, 'std':cstd, 'median':cmedian}
        else : return {}

        
    def distinct(self, colName) :
        res = {}
        for item in self.col(colName) : 
            if item in res : res[item]+=1
            else : res[item]=1
        return res

    def subTable(self, tableName, colName, predFun) :
        ntable = table(tableName) # create a new table
        fltr = predFun(self.col(colName)) # build a filter of rows to retain
        for i,name in zip(range(0,len(self.cols)),self.cols) :
            ncol = self.col(name)[fltr]
            ntable.addCol(name,self.type[i],ncol)
        return ntable

    def find(self,colName,val) :
        """Return a sequence of rows with the entry is specified column equal to val."""
        col = self.col(colName)
        return [self.row(i) for i in range(0,self.len) if col[i] == val]

    def groupTable(self, tableName, colGroupName, colKeepNames) :
        assert self.len > 0, "attempt group on empty table"
        index = np.argsort(self.col(colGroupName))
        scol = self.col(colGroupName)[index] # sorted grouping column

        ranges = getRanges(scol)
        starts = getStarts(ranges)
        groups = scol[starts]

        ntable = table(tableName)
        ntable.addCol(colGroupName, self.type[self.cols.index(colGroupName)], groups)

        for colName in colKeepNames :
            type = self.type[self.cols.index(colName)]
            skcol = self.col(colName)[index] # sorted version of col to keep
            if type == 'STR' : 
               ntable.addCol(colName, type, [getDistinct(skcol[s:e]) for s,e in ranges])
            else :
                ntable.addCol(colName+'-count', 'NUM', [len(skcol[s:e]) for s,e in ranges])
                ntable.addCol(colName+'-min', type, [np.min(skcol[s:e])  for s,e in ranges])
                ntable.addCol(colName+'-mean', type,[np.mean(skcol[s:e]) for s,e in ranges])
                ntable.addCol(colName+'-max', type, [np.max(skcol[s:e])  for s,e in ranges])
                ntable.addCol(colName+'-sum', type, [np.sum(skcol[s:e])  for s,e in ranges])
        return ntable

                
                
            
