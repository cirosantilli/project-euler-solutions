import ProjectEulerStatements.Lib.PowerIntervals
import lang.py.builtin.type.int

namespace ProjectEulerSolutions.PowerIntervals
theorem bitLength_eq_size (value : Nat) : FormalSW.Py.Builtin.Int.bitLength value = value.size := by
  by_cases zero : value = 0
  · simp [zero, FormalSW.Py.Builtin.Int.bitLength]
  · simp only [FormalSW.Py.Builtin.Int.bitLength, Int.natAbs_natCast, if_neg zero]
    have limits := (Nat.log2_eq_iff zero).mp rfl
    have upper := Nat.size_le.mpr limits.2
    have lower := Nat.lt_size.mpr limits.1
    omega

end ProjectEulerSolutions.PowerIntervals
