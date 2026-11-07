import Mathlib.Data.Nat.Bitwise
import lang.py.builtin.type.int

/-! Extracting and removing the least set bit with Python's signed bit operations. -/
namespace ProjectEulerSolutions.LowBit
set_option Elab.async false

def value (n : Nat) : Nat := Nat.ldiff n (n - 1)

@[simp] theorem diff_zero (n : Nat) : Nat.ldiff n 0 = n := by
  apply Nat.eq_of_testBit_eq
  simp

theorem xor_and (n m : Nat) : n ^^^ (n &&& m) = Nat.ldiff n m := by
  apply Nat.eq_of_testBit_eq
  intro index
  simp only [Nat.testBit_xor, Nat.testBit_and, Nat.testBit_ldiff]
  cases n.testBit index <;> cases m.testBit index <;> rfl

theorem integer (n : Nat) : FormalSW.Py.Builtin.Int.bitAnd n (-(n : Int)) = value n := by
  cases n with
  | zero => simp [FormalSW.Py.Builtin.Int.bitAnd, value]
  | succ n =>
      change ((n + 1 ^^^ ((n + 1) &&& n) : Nat) : Int) = _
      rw [xor_and]
      rfl

theorem power (n : Nat) (nonzero : n ≠ 0) :
    ∃ digit, value n = 2 ^ digit ∧ n.testBit digit = true ∧ ∀ index < digit, n.testBit index = false := by
  induction n using Nat.binaryRec with
  | zero => contradiction
  | bit b n ih =>
      cases b with
      | false =>
          have positive : n ≠ 0 := by simpa using nonzero
          obtain ⟨digit, extracted, present, first⟩ := ih positive
          have predecessor : Nat.bit false n - 1 = Nat.bit true (n - 1) := by
            simp only [Nat.bit_false, Nat.bit_true]
            omega
          refine ⟨digit + 1, ?_, by simpa only [Nat.testBit_bit_succ] using present, ?_⟩
          · rw [value, predecessor, Nat.ldiff_bit]
            simp only [Bool.not_true, Bool.and_false, Nat.bit_false, pow_succ]
            change 2 * value n = 2 ^ digit * 2
            rw [extracted, Nat.mul_comm]
          · intro index below
            cases index with
            | zero => simp
            | succ index => simpa only [Nat.testBit_bit_succ] using first index (by omega)
      | true =>
          have predecessor : Nat.bit true n - 1 = Nat.bit false n := by simp
          have self : Nat.ldiff n n = 0 := by apply Nat.eq_of_testBit_eq; simp
          refine ⟨0, ?_, by simp, by intros; omega⟩
          rw [value, predecessor, Nat.ldiff_bit, self]
          rfl

theorem subtract (n digit : Nat) (present : n.testBit digit = true) :
    n - 2 ^ digit = Nat.ldiff n (2 ^ digit) := by
  induction n using Nat.binaryRec generalizing digit with
  | zero => simp at present
  | bit b n ih =>
      cases digit with
      | zero =>
          cases b with
          | false => simp at present
          | true =>
              change Nat.bit true n - 1 = Nat.ldiff (Nat.bit true n) (Nat.bit true 0)
              rw [Nat.ldiff_bit]
              simp
      | succ digit =>
          have member : n.testBit digit = true := by simpa only [Nat.testBit_bit_succ] using present
          have bound : 2 ^ digit ≤ n := by
            by_contra outside
            have zero := Nat.testBit_lt_two_pow (by omega : n < 2 ^ digit)
            simp [member] at zero
          have bitPower : 2 ^ (digit + 1) = Nat.bit false (2 ^ digit) := by simp [pow_succ, Nat.mul_comm]
          rw [bitPower, Nat.ldiff_bit]
          have removed := ih digit member
          cases b <;> simp only [Bool.not_false, Bool.and_true, Nat.bit_true, Nat.bit_false] <;> omega

end ProjectEulerSolutions.LowBit
