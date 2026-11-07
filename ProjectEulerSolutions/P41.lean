import ProjectEulerSolutions.Lib.PandigitalPrimeSearch
namespace ProjectEulerSolutions.P41

/-- Search widths downwards, pruning digit sums and final digits before primality tests. -/
def solve (maxDigits : Nat) : Nat := PandigitalPrimeSearch.search maxDigits
end ProjectEulerSolutions.P41
