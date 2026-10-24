import ProjectEulerSolutions.P19
import ProjectEulerStatements.Lib.Calendar

/-! Equivalence for complete calendar years beginning no earlier than 1900. -/
namespace ProjectEulerSolutions.P19

open ProjectEulerStatements.P19 (dayOfWeek daysBeforeYear daysBeforeMonth
  sundaysInMonths sundaysInYear sundaysInYears firstOfYear lastOfYear)

theorem isLeapYear_eq (year : Nat) : isLeapYear year = ProjectEulerStatements.P19.isLeap year := by
  simpa only [isLeapYear] using (ProjectEulerStatements.P19.isLeap_eq_century_test year).symm

theorem daysInMonth_eq (year month : Nat) (positive : 0 < month) (bound : month ≤ 12) :
    daysInMonth year month = ProjectEulerStatements.P19.daysInMonth year month := by
  interval_cases month <;> simp [daysInMonth, ProjectEulerStatements.P19.daysInMonth, isLeapYear_eq]

private theorem next_year_weekday (year : Nat) (afterEpoch : 1900 ≤ year) :
    dayOfWeek year 13 1 = dayOfWeek (year + 1) 1 1 := by
  simp [dayOfWeek, ProjectEulerStatements.P19.daysBeforeMonth_thirteen,
    ProjectEulerStatements.P19.daysBeforeYear_succ year afterEpoch]

private theorem months_correct (startYear endYear year month count : Nat)
    (afterEpoch : 1900 ≤ year) (active : year ≤ endYear)
    (positive : 0 < month) (bound : month ≤ 13) :
    countSundaysOnFirst.loopYear.loopMonth startYear endYear year month
        (dayOfWeek year month 1) count =
      countSundaysOnFirst.loopYear startYear endYear (year + 1)
        (dayOfWeek (year + 1) 1 1)
        (count + if startYear ≤ year then sundaysInMonths year month (13 - month) else 0) := by
  induction measure : 13 - month using Nat.strong_induction_on generalizing month count with
  | h remaining ih =>
      rw [← measure]
      rw [countSundaysOnFirst.loopYear.loopMonth.eq_1, if_neg (by omega)]
      by_cases finished : month > 12
      · have last : month = 13 := by omega
        subst month
        simp [next_year_weekday year afterEpoch]
      · rw [if_neg finished]
        simp only [daysInMonth_eq year month positive (by omega),
          ← ProjectEulerStatements.P19.firstOfMonth_succ year month positive]
        rw [ih (13 - (month + 1)) (by omega) (month + 1) _ (by omega) (by omega) rfl]
        have length : 13 - month = (13 - (month + 1)) + 1 := by omega
        rw [length, ProjectEulerStatements.P19.sundaysInMonths_succ]
        by_cases eligible : startYear ≤ year <;>
          by_cases sunday : dayOfWeek year month 1 = 6 <;>
          simp [eligible, sunday, active, Nat.add_assoc]

private theorem years_correct (startYear endYear year count : Nat) (afterEpoch : 1900 ≤ year) :
    countSundaysOnFirst.loopYear startYear endYear year (dayOfWeek year 1 1) count =
      count + sundaysInYears (max startYear year) (endYear + 1 - max startYear year) := by
  induction measure : endYear + 1 - year using Nat.strong_induction_on generalizing year count with
  | h remaining ih =>
      rw [countSundaysOnFirst.loopYear.eq_1]
      by_cases finished : year > endYear
      · rw [if_pos finished]
        have empty : endYear + 1 - max startYear year = 0 := by omega
        simp [empty]
      · rw [if_neg finished, months_correct startYear endYear year 1 count afterEpoch
          (by omega) (by decide) (by decide)]
        rw [ih (endYear + 1 - (year + 1)) (by omega) (year + 1) _ (by omega) rfl]
        by_cases eligible : startYear ≤ year
        · simp only [if_pos eligible, Nat.max_eq_right eligible,
            Nat.max_eq_right (show startYear ≤ year + 1 by omega)]
          have length : endYear + 1 - year = (endYear + 1 - (year + 1)) + 1 := by omega
          rw [length, ProjectEulerStatements.P19.sundaysInYears_succ]
          simp [sundaysInYear, Nat.add_assoc]
        · simp [eligible, Nat.max_eq_left (show year ≤ startYear by omega),
            Nat.max_eq_left (show year + 1 ≤ startYear by omega)]

theorem countSundaysOnFirst_eq (startYear endYear : Nat) (afterEpoch : 1900 ≤ startYear) :
    countSundaysOnFirst startYear endYear = sundaysInYears startYear (endYear + 1 - startYear) := by
  have result := years_correct startYear endYear 1900 0 (by decide)
  simpa [countSundaysOnFirst, dayOfWeek, Nat.max_eq_left afterEpoch] using result

/-- The optimized interface counts complete years, not arbitrary endpoint dates. -/
theorem equiv (startYear endYear : Nat) (afterEpoch : 1900 ≤ startYear) (ordered : startYear ≤ endYear) :
    ProjectEulerStatements.P19.naive (firstOfYear startYear) (lastOfYear endYear) =
      solve (firstOfYear startYear) (lastOfYear endYear) := by
  rw [ProjectEulerStatements.P19.naive_whole_years startYear endYear afterEpoch ordered]
  exact (countSundaysOnFirst_eq startYear endYear afterEpoch).symm

end ProjectEulerSolutions.P19
