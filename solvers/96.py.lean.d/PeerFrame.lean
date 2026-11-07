import solvers.«96.py.lean.d».PeerUpdate

namespace solvers.«96».build_peers_units.peerBuild
open FormalSW.PyAst HeapCalls ProjectEulerSolutions ProjectEulerSolutions.Sudoku reduce_cands
set_option Elab.async false
noncomputable section

structure Frame (root : Nat) (before after : Mutable.Heap) : Prop where
  length : before.length ≤ after.length
  outside : ∀ index < before.length, (∀ cell, address root cell ≠ index) → after[index]? = before[index]?

theorem Frame.refl (root : Nat) (heap : Mutable.Heap) : Frame root heap heap :=
  ⟨le_rfl, fun _ _ _ => rfl⟩

theorem Frame.trans {root : Nat} {first middle last : Mutable.Heap}
    (a : Frame root first middle) (b : Frame root middle last) : Frame root first last :=
  ⟨a.length.trans b.length, fun index inside different =>
    (b.outside index (inside.trans_le a.length) different).trans (a.outside index inside different)⟩

theorem Frame.append (root : Nat) (heap extra : Mutable.Heap) : Frame root heap (heap ++ extra) :=
  ⟨by simp, fun _ inside _ => List.getElem?_append_left inside⟩

theorem update_frame (root : Nat) (heap : Mutable.Heap) (keys : Keys) (cell : Cell) (unitKeys : List Constant) :
    Frame root heap (work root heap keys cell unitKeys) :=
  ⟨by rw [length]; omega, fun index inside different => frame root heap keys cell unitKeys index inside (different cell)⟩

theorem View.not_list {root : Nat} {heap : Mutable.Heap} {keys : Keys} (view : View root heap keys)
    (index : Nat) (items : List Mutable.Value) (found : heap[index]? = some (.list items)) :
    ∀ cell, address root cell ≠ index := by
  intro cell same
  have wrong := view.sets cell
  rw [same, found] at wrong
  cases wrong

theorem Frame.list {root : Nat} {before after : Mutable.Heap} {keys : Keys} (frame : Frame root before after)
    (view : View root before keys) (index : Nat) (items : List Mutable.Value) (found : before[index]? = some (.list items)) :
    after[index]? = some (.list items) :=
  (frame.outside index (List.getElem?_eq_some_iff.mp found).1 (view.not_list index items found)).trans found

theorem Frame.units {root : Nat} {before after : Mutable.Heap} {keys : Keys} (frame : Frame root before after)
    (view : View root before keys) (units : UnitTable) (known : units.View before) : units.View after :=
  ⟨frame.list view units.root _ known.root, fun unit member => frame.list view unit.address _ (known.units unit member)⟩

theorem View.fresh {root : Nat} {heap : Mutable.Heap} {keys : Keys} (view : View root heap keys) :
    ∀ cell, address root cell ≠ heap.length := by
  intro cell
  have bound := (List.getElem?_eq_some_iff.mp (view.sets cell)).1
  omega

def unitKeys (cells : Fin 9 → Cell) : List Constant :=
  ((List.finRange 9).map fun index => Peers.atom (cells index)).foldl ScalarSet.insert []

theorem unit_within (cells : Fin 9 → Cell) : Peers.Within cells (unitKeys cells) := Peers.from_unit cells

theorem unit_set (cells : Fin 9 → Cell) (unitAddress : Nat) (env : HeapCalls.Env) (heap : Mutable.Heap)
    (globals : Mutable.Globals) (found : heap[unitAddress]? = some (.list (peerLoop.encoded ((List.finRange 9).map cells))))
    (getUnit : env "u" = some (.ref unitAddress)) :
    Evaluates program env (.builtinCall .set [.name "u"]) heap (.ref heap.length)
      (heap ++ [.set (unitKeys cells)]) globals := by
  apply Evaluates.builtin (.cons (Evaluates.lookup getUnit) .nil)
  apply Mutable.setFrom_scalar_sequence (tuple := false)
  simpa only [Mutable.scalarSequence, Bool.false_eq_true, if_false, peerLoop.encoded,
    List.map_map, Function.comp_def] using found

end
end solvers.«96».build_peers_units.peerBuild
