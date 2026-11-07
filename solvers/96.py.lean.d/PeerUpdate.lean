import solvers.«96.py.lean.d».PeerStorage

namespace solvers.«96».build_peers_units.peerBuild
open FormalSW.PyAst HeapCalls ProjectEulerSolutions ProjectEulerSolutions.Sudoku reduce_cands
set_option Elab.async false
noncomputable section

abbrev difference : Expr := .binOp (.name "s") .sub (.builtinCall .setDisplay [.name "i"])
abbrev statement : Stmt := .augAssign (.subscript (.name "peers") (.name "i")) .bitOr difference
abbrev unitBody : List Stmt := match definition.2[5]? with
  | some (Stmt.forIn _ _ body _) => body
  | _ => []

theorem unit_source : unitBody = [.assign [.name "s"] (.builtinCall .set [.name "u"]),
    .forIn (.name "i") (.name "u") [statement] []] := rfl

def replacement (keys : Keys) (cell : Cell) (unitKeys : List Constant) : List Constant :=
  ScalarSet.union (keys cell) (ScalarSet.difference unitKeys [Peers.atom cell])

def work (root : Nat) (heap : Mutable.Heap) (keys : Keys) (cell : Cell) (unitKeys : List Constant) : Mutable.Heap :=
  (heap ++ [Mutable.Object.set [Peers.atom cell], Mutable.Object.set (ScalarSet.difference unitKeys [Peers.atom cell])]).set
    (address root cell) (.set (replacement keys cell unitKeys))

theorem subtract (cell : Cell) (unitAddress : Nat) (unitKeys : List Constant) (env : HeapCalls.Env)
    (heap : Mutable.Heap) (globals : Mutable.Globals) (found : heap[unitAddress]? = some (.set unitKeys))
    (getSet : env "s" = some (.ref unitAddress)) (getCell : env "i" = some (number (Encoding.index cell).val)) :
    Evaluates program env difference heap (.ref (heap.length + 1))
      (heap ++ [.set [Peers.atom cell], .set (ScalarSet.difference unitKeys [Peers.atom cell])]) globals := by
  have singleton : Evaluates program env (.builtinCall .setDisplay [.name "i"]) heap (.ref heap.length)
      (heap ++ [.set [Peers.atom cell]]) globals := by
    apply Evaluates.builtin (.cons (Evaluates.lookup getCell) .nil)
    exact Mutable.allocateSetValues_scalars heap [Peers.atom cell]
  have result := Evaluates.binary (Evaluates.lookup getSet) singleton
    (Mutable.binary_difference_sets (heap ++ [.set [Peers.atom cell]]) unitAddress heap.length unitKeys [Peers.atom cell]
      (HeapCollection.prefix_found heap _ unitAddress _ found) (by simp))
  simpa only [List.length_append, List.length_singleton, List.append_assoc, List.singleton_append] using result

/-- Augmented subscription retains the peer-set identity and writes it back through the captured list slot. -/
theorem execute (root : Nat) (keys : Keys) (cell : Cell) (unitAddress : Nat) (unitKeys : List Constant)
    (env : HeapCalls.Env) (heap : Mutable.Heap) (globals : Mutable.Globals) (view : View root heap keys)
    (found : heap[unitAddress]? = some (.set unitKeys)) (getPeers : env "peers" = some (.ref root))
    (getSet : env "s" = some (.ref unitAddress)) (getCell : env "i" = some (number (Encoding.index cell).val)) :
    BlockContinues program env [statement] heap env (work root heap keys cell unitKeys) globals := by
  let extended := heap ++ [.set [Peers.atom cell], .set (ScalarSet.difference unitKeys [Peers.atom cell])]
  have extendedView : View root extended keys := view.append _
  have differenceFound : extended[heap.length + 1]? = some (.set (ScalarSet.difference unitKeys [Peers.atom cell])) := by
    simp [extended]
  apply BlockContinues.augItem (afterOperation := work root heap keys cell unitKeys)
    (updated := .ref (address root cell)) (by intros; simp) (Evaluates.lookup getPeers) (Evaluates.lookup getCell)
    (view.read cell) (subtract cell unitAddress unitKeys env heap globals found getSet getCell)
    (inplace_union_sets extended (address root cell) (heap.length + 1) (keys cell)
      (ScalarSet.difference unitKeys [Peers.atom cell]) (extendedView.sets cell) differenceFound)
  exact (extendedView.update cell (replacement keys cell unitKeys)).write_same cell

theorem view (root : Nat) (heap : Mutable.Heap) (keys : Keys) (cell : Cell) (unitKeys : List Constant)
    (before : View root heap keys) :
    View root (work root heap keys cell unitKeys) (Function.update keys cell (replacement keys cell unitKeys)) :=
  (before.append _).update cell _

theorem length (root : Nat) (heap : Mutable.Heap) (keys : Keys) (cell : Cell) (unitKeys : List Constant) :
    (work root heap keys cell unitKeys).length = heap.length + 2 := by simp [work]

theorem frame (root : Nat) (heap : Mutable.Heap) (keys : Keys) (cell : Cell) (unitKeys : List Constant)
    (index : Nat) (inside : index < heap.length) (different : address root cell ≠ index) :
    (work root heap keys cell unitKeys)[index]? = heap[index]? := by
  rw [work, List.getElem?_set_ne different, List.getElem?_append_left inside]

theorem related (keys : Keys) (unit : Fin 9 → Cell) (unitKeys : List Constant) (index : Fin 9)
    (before : ∀ cell, Peers.Related cell (keys cell)) (within : Peers.Within unit unitKeys)
    (unitRelated : ∀ i j, i ≠ j → Peer (unit i) (unit j)) :
    ∀ cell, Peers.Related cell (Function.update keys (unit index) (replacement keys (unit index) unitKeys) cell) := by
  intro cell
  by_cases same : cell = unit index
  · subst cell
    simp only [Function.update_self]
    exact Peers.update _ _ _ (before _) (Peers.without_self unit unitRelated unitKeys within index)
  · simpa only [Function.update_of_ne same] using before cell

end
end solvers.«96».build_peers_units.peerBuild
