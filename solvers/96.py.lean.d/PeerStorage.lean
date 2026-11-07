import solvers.«96.py.lean.d».UnitConstruction
import ProjectEulerSolutions.Lib.SudokuPeers
import lang.py.FormalSW.SetDifferenceCalls

namespace solvers.«96».build_peers_units.peerBuild
open FormalSW.PyAst HeapCalls ProjectEulerSolutions ProjectEulerSolutions.Sudoku reduce_cands
set_option Elab.async false
noncomputable section

def address (root : Nat) (cell : Cell) : Nat := root + 1 + (Encoding.index cell).val
abbrev Keys := Cell → List Constant

structure View (rootAddress : Nat) (heap : Mutable.Heap) (keys : Keys) : Prop where
  root : heap[rootAddress]? = some (.list (references rootAddress 81))
  sets : ∀ cell, heap[address rootAddress cell]? = some (.set (keys cell))

theorem View.append {root : Nat} {heap : Mutable.Heap} {keys : Keys} (view : View root heap keys)
    (extra : Mutable.Heap) : View root (heap ++ extra) keys :=
  ⟨HeapCollection.prefix_found heap extra _ _ view.root,
   fun cell => HeapCollection.prefix_found heap extra _ _ (view.sets cell)⟩

theorem addresses_injective (root : Nat) : Function.Injective (address root) := by
  intro a b same
  have indices : Encoding.index a = Encoding.index b := Fin.ext (by simpa [address] using same)
  simpa only [Encoding.cell_index] using congrArg Encoding.cell indices

theorem View.update {root : Nat} {heap : Mutable.Heap} {keys : Keys} (view : View root heap keys)
    (cell : Cell) (replacement : List Constant) :
    View root (heap.set (address root cell) (.set replacement)) (Function.update keys cell replacement) := by
  constructor
  · rw [List.getElem?_set_ne (by unfold address; omega)]
    exact view.root
  · intro other
    by_cases same : other = cell
    · subst other
      simp [Function.update_self, (List.getElem?_eq_some_iff.mp (view.sets cell)).1]
    · rw [List.getElem?_set_ne (fun equal => same ((addresses_injective root equal).symm))]
      simpa only [Function.update_of_ne same] using view.sets other

theorem View.read {root : Nat} {heap : Mutable.Heap} {keys : Keys} (view : View root heap keys) (cell : Cell) :
    Mutable.getItem (.ref root) (number (Encoding.index cell).val) heap = some (.ok (.ref (address root cell)), heap) := by
  simp [Mutable.getItem, view.root, Mutable.index, references, address,
    not_lt.mpr (Int.natCast_nonneg (Encoding.index cell).val), not_le.mpr (Encoding.index cell).isLt, Except.map]

theorem View.write_same {root : Nat} {heap : Mutable.Heap} {keys : Keys} (view : View root heap keys) (cell : Cell) :
    Mutable.setItem (.ref root) (number (Encoding.index cell).val) (.ref (address root cell)) heap =
      some (.ok (.scalar .none), heap) := by
  have index : (references root 81)[(Encoding.index cell).val]? = some (.ref (address root cell)) := by
    simp [references, address, (Encoding.index cell).isLt]
  obtain ⟨_, selected⟩ := List.getElem_of_getElem? index
  have unchanged : (references root 81).set (Encoding.index cell).val (.ref (address root cell)) = references root 81 := by
    rw [← selected, List.set_getElem_self]
  have heapUnchanged : heap.set root (.list (references root 81)) = heap := by
    obtain ⟨inside, entry⟩ := List.getElem?_eq_some_iff.mp view.root
    rw [← entry, List.set_getElem_self inside]
  simp [Mutable.setItem, view.root, Mutable.index, show (references root 81).length = 81 by simp [references],
    not_lt.mpr (Int.natCast_nonneg (Encoding.index cell).val), not_le.mpr (Encoding.index cell).isLt,
    unchanged, heapUnchanged, Except.map]

def emptyWork (base : Mutable.Heap) (count : Nat) : Mutable.Heap :=
  HeapCollection.working base (references base.length count) (List.replicate count (.set []))

abbrev collection : Expr := .listComp (.builtinCall .set []) [("_", .call (.name "range") [.constant (.int 81)], [])]

theorem source : definition.2[4]? = some (.assign [.name "peers"] collection) := rfl

theorem allocate (env : HeapCalls.Env) (heap : Mutable.Heap) (globals : Mutable.Globals) :
    Evaluates program env collection heap (.ref heap.length) (emptyWork heap 81) globals := by
  let update := fun count (_ : Fin 81) => count + 1
  have callback (count : Nat) (scope : HeapCalls.Env) (index : Fin 81) (_ : True)
      (_ : scope "_" = some (.scalar (.int (0 + 1 * index.val)))) :
      ∃ afterEnv, BlockContinues program scope [.comprehensionAppend heap.length (.builtinCall .set [])]
        (emptyWork heap count) afterEnv (emptyWork heap (count + 1)) globals ∧ True := by
    have created : Evaluates program scope (.builtinCall .set []) (emptyWork heap count)
        (.ref (emptyWork heap count).length) (emptyWork heap count ++ [.set []]) globals :=
      Evaluates.builtin .nil rfl
    have appended := BlockContinues.comprehension_append created
      (HeapCollection.prefix_found (emptyWork heap count) _ heap.length _ (HeapCollection.found heap _ _))
    refine ⟨scope, ?_, trivial⟩
    simpa [emptyWork, HeapCollection.working, references, List.range_succ, List.set_append_right,
      List.append_assoc, List.replicate_succ', Nat.add_assoc, Nat.add_comm, Nat.add_left_comm] using appended
  obtain ⟨afterEnv, executed, _⟩ := HeapFiniteRange.loop program "_"
    [.comprehensionAppend heap.length (.builtinCall .set [])] rfl 81 0 1 (by decide)
    (emptyWork heap) update (fun _ => True) globals (by intros; trivial) callback
    (List.finRange 81) 0 rfl (by omega) 0 (comprehensionEnv env [("_", .call (.name "range") [.constant (.int 81)], [])]) trivial
  apply Evaluates.list_comp_range (finalEnv := afterEnv) (.cons Evaluates.constant .nil) rfl
  exact executed

theorem empty_view (heap : Mutable.Heap) : View heap.length (emptyWork heap 81) (fun _ => []) := by
  constructor
  · exact HeapCollection.found _ _ _
  · intro cell
    have offset : address heap.length cell - heap.length = (Encoding.index cell).val + 1 := by unfold address; omega
    unfold emptyWork HeapCollection.working
    rw [List.getElem?_append_right (show heap.length ≤ address heap.length cell by unfold address; omega), offset]
    exact List.getElem?_replicate_of_lt (Encoding.index cell).isLt

end
end solvers.«96».build_peers_units.peerBuild
