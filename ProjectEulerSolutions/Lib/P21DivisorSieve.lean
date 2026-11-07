import ProjectEulerStatements.Lib.DivisorSieve

/-! An executable array proper-divisor sieve using the shared partial-sum invariant. -/
namespace ProjectEulerSolutions.P21DivisorSieve
open ProjectEulerStatements.DivisorSieve
set_option Elab.async false

def encode (size : Nat) (value : Nat → Nat) : Array Nat := Array.ofFn (fun i : Fin size => value i)
@[simp] theorem encode_size (size : Nat) (value : Nat → Nat) : (encode size value).size = size := by simp [encode]

def add (cells : Array Nat) (position amount : Nat) : Array Nat :=
  if inside : position < cells.size then cells.set position (cells[position]+amount) else cells

theorem add_encode (size position amount : Nat) (value : Nat → Nat) (inside : position < size) :
    add (encode size value) position amount =
      encode size (fun n => if n = position then value n + amount else value n) := by
  apply Array.ext
  · simp [add, inside]
  · intro n hl hr
    simp only [add, encode_size, dif_pos inside, Array.getElem_set]
    by_cases same : position = n
    · subst n; simp [encode]
    · simp [encode, same, Ne.symm same]

theorem encode_congr (size : Nat) (f g : Nat → Nat) (same : ∀ n, n < size → f n = g n) :
    encode size f = encode size g := by
  apply Array.ext
  · simp
  · intro n hl hr
    simp only [encode, Array.getElem_ofFn]
    exact same n (by simpa using hl)

def multiples (maximum step position : Nat) (positive : 0 < step) (cells : Array Nat) : Array Nat :=
  if position ≤ maximum then multiples maximum step (position+step) positive (add cells position step)
  else cells
termination_by maximum+1-position
decreasing_by omega

theorem multiples_correct (maximum step position : Nat) (positive : 0 < step)
    (lower : 2*step ≤ position) (multiple : step ∣ position) :
    multiples maximum step position positive (encode (maximum+1) (row (sumsBefore step) step position)) =
      encode (maximum+1) (sumsBefore (step+1)) := by
  rw [multiples]
  split
  · rename_i inside
    rw [add_encode _ _ _ _ (by omega)]
    have update : (fun n => if n = position then row (sumsBefore step) step position n + step
        else row (sumsBefore step) step position n) = row (sumsBefore step) step (position+step) := by
      funext n
      exact (row_step _ _ _ _ positive lower multiple).symm
    rw [update]
    apply multiples_correct maximum step (position+step) positive (by omega) (Nat.dvd_add multiple (dvd_refl step))
  · rename_i done
    apply encode_congr
    intro n hn
    exact row_complete step position n positive (by omega)
termination_by maximum+1-position
decreasing_by omega

def outer (maximum step : Nat) (positive : 0 < step) (cells : Array Nat) : Array Nat :=
  if step ≤ maximum / 2 then
    outer maximum (step+1) (by omega) (multiples maximum step (2*step) positive cells)
  else cells
termination_by maximum/2+1-step
decreasing_by omega

theorem outer_correct (maximum step : Nat) (positive : 0 < step) :
    outer maximum step positive (encode (maximum+1) (sumsBefore step)) =
      encode (maximum+1) ProjectEulerStatements.P21.d := by
  rw [outer]
  split
  · have pass : multiples maximum step (2*step) positive (encode (maximum+1) (sumsBefore step)) =
        encode (maximum+1) (sumsBefore (step+1)) := by
      have start : row (sumsBefore step) step (2*step) = sumsBefore step := by
        funext n; exact row_start _ _ _
      rw [← start]
      exact multiples_correct maximum step (2*step) positive (by omega) (dvd_mul_left step 2)
    rw [pass]
    exact outer_correct maximum (step+1) (by omega)
  · rename_i done
    apply encode_congr
    intro n hn
    apply sumsBefore_complete
    omega
termination_by maximum/2+1-step
decreasing_by omega

def compute (maximum : Nat) : Array Nat := outer maximum 1 (by decide) (Array.replicate (maximum+1) 0)

theorem compute_correct (maximum : Nat) : compute maximum = encode (maximum+1) ProjectEulerStatements.P21.d := by
  have initial : Array.replicate (maximum+1) 0 = encode (maximum+1) (sumsBefore 1) := by
    apply Array.ext
    · simp
    · intro n hl hr; simp [encode]
  rw [compute, initial, outer_correct]
end ProjectEulerSolutions.P21DivisorSieve
