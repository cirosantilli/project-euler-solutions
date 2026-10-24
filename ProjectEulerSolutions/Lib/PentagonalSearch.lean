import ProjectEulerSolutions.Lib.PentagonalDiscriminant
import Mathlib.Data.List.Fold

/-! A bounded minimum over pentagonal pairs, including the descending-search pruning rule. -/
namespace ProjectEulerSolutions.PentagonalSearch
open PentagonalDiscriminant

set_option Elab.async false

def cap : Nat := 10^30
def difference (k j : Nat) : Nat := pentagonal k - pentagonal j
def accepted (k j : Nat) : Bool :=
  exhaustive (pentagonal k + pentagonal j) && exhaustive (difference k j)
def rowValues (k count : Nat) : List Nat :=
  ((List.range' 1 count).filter (accepted k)).map (difference k)
def row (k count best : Nat) : Nat := (rowValues k count).foldl min best
def values (limit : Nat) : List Nat := (List.range' 1 limit).flatMap fun k => rowValues k (k-1)
def search (limit : Nat) : Nat := (values limit).foldl min cap

private theorem fold_min_insert (items : List Nat) (initial candidate : Nat) :
    items.foldl min (min initial candidate) = min (items.foldl min initial) candidate := by
  induction items generalizing initial with
  | nil => rfl
  | cons first rest ih =>
      simp only [List.foldl_cons]
      rw [show min (min initial candidate) first = min (min initial first) candidate by omega, ih]

/-- Scanning a row backwards changes neither its set of candidates nor the bounded minimum. -/
theorem row_succ (k count best : Nat) : row k (count+1) best =
    row k count (if accepted k (count+1) then min best (difference k (count+1)) else best) := by
  have enumeration : List.range' 1 (count+1) = List.range' 1 count ++ [count+1] := by
    simpa only [Nat.add_comm] using (List.range'_1_concat (s := 1) (n := count))
  simp only [row, rowValues, enumeration, List.filter_append, List.map_append, List.foldl_append]
  cases keep : accepted k (count+1) <;> simp [keep, fold_min_insert]

theorem row_fixed (k count best : Nat)
    (bounded : ∀ j, 0 < j → j ≤ count → best ≤ difference k j) : row k count best = best := by
  induction count with
  | zero => rfl
  | succ count ih =>
      rw [row_succ]
      have current := bounded (count+1) (by omega) (by omega)
      have fixed : (if accepted k (count+1) then min best (difference k (count+1)) else best) = best := by
        split <;> omega
      rw [fixed]
      exact ih (fun j positive upper => bounded j positive (by omega))

/-- Smaller second indices have greater differences, making the Python early break sound. -/
theorem prune (k count best : Nat) (cut : best ≤ difference k count) : row k count best = best := by
  apply row_fixed
  intro j _ bound
  exact cut.trans (Nat.sub_le_sub_left (strictMono.monotone bound) _)

theorem row_nil (k best : Nat) : row k 0 best = best := rfl

theorem values_fold (indices : List Nat) (best : Nat) :
    (indices.flatMap fun k => rowValues k (k-1)).foldl min best =
      indices.foldl (fun previous k => row k (k-1) previous) best := by
  induction indices generalizing best with
  | nil => rfl
  | cons first rest ih =>
      simp only [List.flatMap_cons, List.foldl_append, List.foldl_cons]
      exact ih _

theorem skip_first (limit : Nat) : search limit =
    (List.range' 2 (limit-1)).foldl (fun previous k => row k (k-1) previous) cap := by
  rw [search, values, values_fold]
  cases limit with
  | zero => rfl
  | succ n => simp only [List.range'_succ, List.foldl_cons, Nat.succ_sub_one, Nat.sub_self, row_nil]

end ProjectEulerSolutions.PentagonalSearch
