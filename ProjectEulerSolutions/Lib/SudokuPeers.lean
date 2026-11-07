import ProjectEulerSolutions.Lib.SudokuUnits
import ProjectEulerSolutions.Lib.SudokuEncoding
import lang.py.FormalSW.ScalarSets

/-! The peer builder's scalar-set operations introduce only cells in a shared unit. -/
namespace ProjectEulerSolutions.Sudoku.Peers
open FormalSW.PyAst
set_option Elab.async false
noncomputable section

abbrev atom (cell : Cell) : Constant := .int (Encoding.index cell).val

def Within (unit : Fin 9 → Cell) (keys : List Constant) : Prop :=
  ∀ key ∈ keys, ∃ index, key = atom (unit index)

def Related (cell : Cell) (keys : List Constant) : Prop :=
  ∀ key ∈ keys, ∃ other, key = atom other ∧ Peer cell other

theorem all_insert (predicate : Constant → Prop) (keys : List Constant) (item : Constant)
    (before : ∀ key ∈ keys, predicate key) (added : predicate item) :
    ∀ key ∈ ScalarSet.insert keys item, predicate key := by
  unfold ScalarSet.insert
  split
  · exact before
  · intro key member
    rcases List.mem_append.mp member with old | new
    · exact before key old
    · simpa only [List.mem_singleton.mp new] using added

theorem all_union (predicate : Constant → Prop) (left right : List Constant)
    (before : ∀ key ∈ left, predicate key) (added : ∀ key ∈ right, predicate key) :
    ∀ key ∈ ScalarSet.union left right, predicate key := by
  induction right generalizing left with
  | nil => exact before
  | cons item rest ih =>
      exact ih (ScalarSet.insert left item) (all_insert predicate left item before (added item (by simp)))
        (fun key member => added key (List.mem_cons_of_mem _ member))

theorem from_unit (unit : Fin 9 → Cell) :
    Within unit (((List.finRange 9).map fun index => atom (unit index)).foldl ScalarSet.insert []) := by
  apply all_union (fun key => ∃ index, key = atom (unit index)) [] _ (by simp)
  intro key member
  obtain ⟨index, _, same⟩ := List.mem_map.mp member
  exact ⟨index, same.symm⟩

theorem without_self (unit : Fin 9 → Cell) (related : ∀ i j, i ≠ j → Peer (unit i) (unit j))
    (keys : List Constant) (known : Within unit keys) (index : Fin 9) :
    Related (unit index) (ScalarSet.difference keys [atom (unit index)]) := by
  intro key member
  obtain ⟨present, absent⟩ := (ScalarSet.mem_difference keys [atom (unit index)] key).mp member
  obtain ⟨other, rfl⟩ := known key present
  refine ⟨unit other, rfl, related index other ?_⟩
  intro same
  subst other
  simp [ScalarSet.contains, scalarCompare, evalCmpOp] at absent

theorem update (cell : Cell) (before added : List Constant)
    (old : Related cell before) (new : Related cell added) : Related cell (ScalarSet.union before added) :=
  all_union (fun key => ∃ other, key = atom other ∧ Peer cell other) before added old new

/-- The stored integer representatives recover an actual cell list, preserving their order. -/
theorem decode (cell : Cell) (keys : List Constant) (known : Related cell keys) :
    ∃ cells : List Cell, keys = cells.map atom ∧ ∀ other ∈ cells, Peer cell other := by
  induction keys with
  | nil => exact ⟨[], rfl, by simp⟩
  | cons key rest ih =>
      obtain ⟨other, rfl, peer⟩ := known key (by simp)
      obtain ⟨cells, same, peers⟩ := ih (fun key member => known key (List.mem_cons_of_mem _ member))
      refine ⟨other :: cells, by simp [same], ?_⟩
      intro current member
      rcases List.mem_cons.mp member with rfl | member
      · exact peer
      · exact peers current member

end
end ProjectEulerSolutions.Sudoku.Peers
