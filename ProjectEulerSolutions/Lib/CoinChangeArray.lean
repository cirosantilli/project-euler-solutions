import ProjectEulerStatements.Lib.CoinChange
import Mathlib.Data.List.Range

/-! A dynamically sized coin-change table with proved in-range in-place updates. -/
namespace ProjectEulerSolutions.CoinChangeArray
open ProjectEulerStatements.P31 ProjectEulerStatements.CoinChange
set_option Elab.async false

def encode (size : Nat) (value : Nat → Nat) : Array Nat :=
  Array.ofFn (fun index : Fin size => value index)

@[simp] theorem encode_size (size : Nat) (value : Nat → Nat) : (encode size value).size = size := by
  simp [encode]

def update (cells : Array Nat) (part position : Nat) : Array Nat :=
  if inside : position < cells.size then
    cells.set position (cells[position] + cells[position-part]'(by omega))
  else cells

theorem update_encode (size part position : Nat) (value : Nat → Nat)
    (inside : position < size) :
    update (encode size value) part position = encode size
      (fun query => if query = position then value position + value (position-part)
        else value query) := by
  apply Array.ext
  · simp [update, inside]
  · intro query leftBound rightBound
    simp only [update, encode_size, dif_pos inside, Array.getElem_set]
    by_cases same : position = query
    · subst query
      simp [encode]
    · simp [encode, same, Ne.symm same]

def pass (target part first : Nat) (cells : Array Nat) : Array Nat :=
  if first ≤ target then pass target part (first+1) (update cells part first) else cells
termination_by target+1-first
decreasing_by omega

theorem pass_encode (target part first : Nat) (processed : List Nat)
    (positive : 0 < part) (started : part ≤ first) :
    pass target part first (encode (target+1) (frontier part processed first)) =
      encode (target+1) (fun amount => countWaysBound amount (part :: processed)) := by
  by_cases inside : first ≤ target
  · rw [pass, if_pos inside, update_encode _ _ _ _ (by omega)]
    have step : (fun query => if query = first then frontier part processed first first +
        frontier part processed first (first-part) else frontier part processed first query) =
        frontier part processed (first+1) := by
      funext query
      exact (frontier_step part processed first query positive started).symm
    rw [step]
    exact pass_encode target part (first+1) processed positive (by omega)
  · rw [pass, if_neg inside]
    apply Array.ext
    · simp
    · intro index leftBound rightBound
      have bounded : index < target+1 := by simpa only [encode_size] using leftBound
      simp [encode, frontier, show index < first by omega]
termination_by target+1-first
decreasing_by omega

theorem pass_start (target part : Nat) (processed : List Nat) (positive : 0 < part) :
    pass target part part (encode (target+1) (fun amount => countWaysBound amount processed)) =
      encode (target+1) (fun amount => countWaysBound amount (part :: processed)) := by
  have start : frontier part processed part = fun amount => countWaysBound amount processed :=
    funext (fun amount => frontier_start part processed amount positive)
  rw [← start]
  exact pass_encode target part part processed positive (by omega)

def process (target : Nat) : List Nat → Array Nat → Array Nat
  | [], cells => cells
  | part :: rest, cells => process target rest (pass target part part cells)

theorem process_encode (target : Nat) (parts processed : List Nat)
    (positive : ∀ part ∈ parts, 0 < part) :
    process target parts (encode (target+1) (fun amount => countWaysBound amount processed)) =
      encode (target+1) (fun amount => countWaysBound amount (parts.reverse ++ processed)) := by
  induction parts generalizing processed with
  | nil => simp [process]
  | cons part rest ih =>
      rw [process, pass_start _ _ _ (positive part (by simp)),
        ih (part :: processed) (fun p member => positive p (by simp [member]))]
      simp [List.reverse_cons, List.append_assoc]

def initialized (target : Nat) : Array Nat :=
  (Array.replicate (target+1) 0).set 0 1 (by simp)

theorem initialized_encode (target : Nat) :
    initialized target = encode (target+1) (fun amount => countWaysBound amount []) := by
  apply Array.ext
  · simp [initialized]
  · intro index leftBound rightBound
    simp [initialized, encode, Array.getElem_set, eq_comm]

def compute (target : Nat) (parts : List Nat) : Array Nat :=
  process target parts (initialized target)

theorem compute_encode (target : Nat) (parts : List Nat)
    (positive : ∀ part ∈ parts, 0 < part) :
    compute target parts =
      encode (target+1) (fun amount => countWaysBound amount parts.reverse) := by
  rw [compute, initialized_encode, process_encode _ _ _ positive]
  simp

end ProjectEulerSolutions.CoinChangeArray
