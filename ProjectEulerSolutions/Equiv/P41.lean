import ProjectEulerSolutions.P41
namespace ProjectEulerSolutions.P41

/-- `through` is the shared model of Python's `naive_bound`: all widths up to the input. -/
theorem equiv (maxDigits : Nat) (decimal : maxDigits ≤ 9) :
    PandigitalPrimeSearch.through maxDigits = solve maxDigits :=
  (PandigitalPrimeSearch.search_correct maxDigits decimal).symm
end ProjectEulerSolutions.P41
