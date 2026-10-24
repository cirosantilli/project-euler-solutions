import ProjectEulerSolutions.Lib.PrimePairCertificate

/-! Generated bounded vertex intervals; every list is kernel checked. -/
namespace ProjectEulerSolutions.PrimePairVertices
open PrimePairCertificate

def interval_8194 : List Nat := [8209, 8219, 8221, 8231, 8233, 8237, 8243, 8263, 8269, 8273, 8287, 8291, 8293, 8297, 8311, 8317]
theorem interval_8194_checked : primeSegment 8194 128 = interval_8194 := by
  rw [← computedSegment_eq_primeSegment 8194 128 (by decide)]
  decide

def interval_8322 : List Nat := [8329, 8353, 8363, 8369, 8377, 8387]
theorem interval_8322_checked : primeSegment 8322 67 = interval_8322 := by
  rw [← computedSegment_eq_primeSegment 8322 67 (by decide)]
  decide

def block_004 : List Nat := interval_8194 ++ interval_8322
theorem block_004_checked : primeSegment 8194 195 = block_004 := by
  change primeSegment 8194 (128 + 67) = _
  rw [primeSegment_append, interval_8194_checked, interval_8322_checked]
  rfl

end ProjectEulerSolutions.PrimePairVertices
