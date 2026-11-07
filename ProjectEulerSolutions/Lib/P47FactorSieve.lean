import ProjectEulerSolutions.Lib.PrimeFactorSieve
import ProjectEulerSolutions.Lib.P21DivisorSieve

/-! In-place prime-factor counting with the same partial-count invariant as the Python sieve. -/
namespace ProjectEulerSolutions.P47FactorSieve
open PrimeFactorSieve P21DivisorSieve
set_option Elab.async false

def multiples (maximum prime position : Nat) (positive : 0 < prime) (cells : Array Nat) : Array Nat :=
  if position ≤ maximum then multiples maximum prime (position+prime) positive (add cells position 1)
  else cells
termination_by maximum+1-position
decreasing_by all_goals omega

theorem multiples_correct (maximum p position : Nat) (prime : Nat.Prime p)
    (lower : p ≤ position) (multiple : p ∣ position) :
    multiples maximum p position prime.pos (encode (maximum+1) (row (countsBefore p) p position)) =
      encode (maximum+1) (countsBefore (p+1)) := by
  rw [multiples]
  split
  · rename_i inside
    rw [add_encode _ _ _ _ (by omega)]
    have update : (fun n => if n = position then row (countsBefore p) p position n + 1
        else row (countsBefore p) p position n) = row (countsBefore p) p (position+p) := by
      funext n
      exact (row_step _ _ _ _ prime.pos lower multiple).symm
    rw [update]
    exact multiples_correct maximum p (position+p) prime (by omega) (Nat.dvd_add multiple (dvd_refl p))
  · rename_i done
    apply encode_congr
    intro n hn
    exact row_complete p position n prime (by omega)
termination_by maximum+1-position
decreasing_by have := prime.pos; omega

def outer (maximum p : Nat) (positive : 2 ≤ p) (cells : Array Nat) : Array Nat :=
  if p ≤ maximum then
    let updated := if cells[p]! = 0 then multiples maximum p p (by omega) cells else cells
    outer maximum (p+1) (by omega) updated
  else cells
termination_by maximum+1-p
decreasing_by all_goals omega

theorem outer_correct (maximum p : Nat) (positive : 2 ≤ p) :
    outer maximum p positive (encode (maximum+1) (countsBefore p)) =
      encode (maximum+1) ProjectEulerStatements.P47.numDistinctPrimeFactors := by
  rw [outer]
  split
  · rename_i inside
    have get : (encode (maximum+1) (countsBefore p))[p]! = countsBefore p p := by
      simp [encode, show p < maximum+1 by omega]
    simp only [get]
    split
    · rename_i zero
      have prime := (current_zero p positive).mp zero
      have start : row (countsBefore p) p p = countsBefore p := by
        funext n; exact row_start _ _ _
      have pass := multiples_correct maximum p p prime (Nat.le_refl _) (dvd_refl p)
      rw [start] at pass
      rw [pass]
      exact outer_correct maximum (p+1) (by omega)
    · rename_i nonzero
      have composite : ¬ Nat.Prime p := fun prime => nonzero ((current_zero p positive).mpr prime)
      have unchanged : countsBefore p = countsBefore (p+1) := by
        funext n; exact skip_composite p n composite
      rw [unchanged]
      exact outer_correct maximum (p+1) (by omega)
  · rename_i done
    apply encode_congr
    intro n hn
    exact countsBefore_complete p n (by omega)
termination_by maximum+1-p
decreasing_by all_goals omega

def compute (maximum : Nat) : Array Nat := outer maximum 2 (by decide) (Array.replicate (maximum+1) 0)

theorem compute_correct (maximum : Nat) :
    compute maximum = encode (maximum+1) ProjectEulerStatements.P47.numDistinctPrimeFactors := by
  have initial : Array.replicate (maximum+1) 0 = encode (maximum+1) (countsBefore 2) := by
    apply Array.ext
    · simp
    · intro n hl hr; simp [encode]
  rw [compute, initial, outer_correct]
end ProjectEulerSolutions.P47FactorSieve
