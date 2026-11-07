import ProjectEulerSolutions.P48
import Mathlib.Data.Nat.ModEq

namespace ProjectEulerSolutions.P48

private theorem go_mod (a b modulus initial : Nat) :
    ProjectEulerStatements.P48.modPow.go modulus a b initial % modulus =
      initial * a ^ b % modulus := by
  induction b generalizing initial with
  | zero => simp [ProjectEulerStatements.P48.modPow.go]
  | succ b ih =>
      rw [ProjectEulerStatements.P48.modPow.go, ih]
      rw [Nat.mod_mul_mod]
      simp only [pow_succ', Nat.mul_assoc]

private theorem modPow_mod (a b modulus : Nat) :
    ProjectEulerStatements.P48.modPow a b modulus % modulus = powMod a b modulus := by
  unfold ProjectEulerStatements.P48.modPow powMod
  rw [go_mod, Nat.one_mul]
  exact Nat.ModEq.pow b (by simp [Nat.ModEq])

theorem equiv (n : Nat) : ProjectEulerStatements.P48.naive n = solve n := by
  unfold ProjectEulerStatements.P48.naive ProjectEulerStatements.P48.seriesMod solve selfPowersMod
  congr 1
  funext total i
  rw [Nat.add_mod, modPow_mod]
  conv_rhs => rw [Nat.add_mod]
  simp [powMod]

end ProjectEulerSolutions.P48
