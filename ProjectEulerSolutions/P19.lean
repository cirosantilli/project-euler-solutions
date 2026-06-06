import ProjectEulerStatements.P19
import ProjectEulerSolutions.Termination.P19
namespace ProjectEulerSolutions.P19

def isLeapYear (year : Nat) : Bool :=
  if year % 400 == 0 then
    true
  else if year % 100 == 0 then
    false
  else
    year % 4 == 0

termination_by 0
decreasing_by all_goals exact Termination.decreases
def daysInMonth (year month : Nat) : Nat :=
  if month == 2 then
    if isLeapYear year then 29 else 28
  else if month == 4 || month == 6 || month == 9 || month == 11 then
    30
  else
    31

termination_by 0
decreasing_by all_goals exact Termination.decreases
def countSundaysOnFirst (startYear endYear : Nat) : Nat :=
  let rec loopYear (year dow count : Nat) : Nat :=
    if year > endYear then
      count
    else
      let rec loopMonth (month dow count : Nat) : Nat :=
        if month > 12 then
          loopYear (year + 1) dow count
        else
          let count' :=
            if startYear <= year && year <= endYear && dow == 6 then
              count + 1
            else
              count
          let dow' := (dow + daysInMonth year month) % 7
          loopMonth (month + 1) dow' count'
      termination_by 0
      decreasing_by all_goals exact Termination.decreases
      loopMonth 1 dow count
  termination_by 0
  decreasing_by all_goals exact Termination.decreases
  loopYear 1900 0 0

termination_by 0
decreasing_by all_goals exact Termination.decreases
def solve (startDate endDate : ProjectEulerStatements.P19.Date) : Nat :=
  countSundaysOnFirst startDate.year endDate.year

example : countSundaysOnFirst 1901 2000 = 171 := by native_decide
end ProjectEulerSolutions.P19
