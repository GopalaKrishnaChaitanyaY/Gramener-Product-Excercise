# 1. Given two lists L1 = ['a', 'b', 'c'], L2 = ['b', 'd'], find common elements, find elements present in L1 and not in L2?

#  Solution
   #Printing Common elements 
L1 = ['a', 'b', 'c']
L2 = ['b', 'd']
print list(set(L1).intersection(set(L2)))
               # Or
print list(set(L1) & (set(L2)))

   #Printing elements in L1 not in L2
print [x for x in L1 if x not in L2]
             #or
print filter(lambda x: x not in L2, L1)

# 2. How many Thursdays were there between 1990 - 2000?
from datetime import datetime
from dateutil import rrule

print len(list(rrule.rrule(rrule.DAILY,
                         dtstart=datetime(1990, 1, 1),
                         until=datetime(2000, 12, 31),
                         byweekday=[rrule.TH])))



from datetime import date
import datetime
import calendar
d0 = date(1990, 1, 1)
d1 = date(2000, 12, 31)
delta = d0 - d1
print delta.days
print list(calendar.day_abbr)
print calendar.day_abbr[3] 

from datetime import datetime
sum(datetime(year, month, 1).weekday() == 4
      for year in range(1950, 2051) for month in range(1,13))