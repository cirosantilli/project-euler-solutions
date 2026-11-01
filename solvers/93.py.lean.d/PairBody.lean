import solvers.«93.py.lean.d».ValueLoop

namespace solvers.«93».all_results
open FormalSW.PyAst HeapCalls ProjectEulerSolutions.RationalPair
open ProjectEulerSolutions ProjectEulerSolutions.IndexedFilter
set_option Elab.async false

abbrev pairCallExpr : Expr := .call (.name "pair_results")
  [.subscript (.name "state") (.name "i"), .subscript (.name "state") (.name "j")]
abbrev pairBody : List Stmt :=
  [.assign [.name "rest"] remainderExpr, .forIn (.name "value") pairCallExpr [.expr updateExpr] []]

private def loopBody : Option Stmt → List Stmt
  | some (.forIn _ _ body _) => body
  | _ => []

theorem source_pair_body : loopBody ((loopBody recBody[3]?)[0]?) = pairBody := rfl

theorem state_item (heap : Mutable.Heap) (state index : Nat) (values : List Mutable.Value)
    (name : String) (env : HeapCalls.Env) (globals : Mutable.Globals)
    (found : heap[state]? = some (.tuple values)) (inside : index < values.length)
    (getState : env "state" = some (.ref state)) (getIndex : env name = some (.scalar (.int index))) :
    Evaluates program env (.subscript (.name "state") (.name name)) heap values[index] heap globals := by
  apply Evaluates.subscript (by intro a b c impossible; cases impossible) (Evaluates.lookup getState) (Evaluates.lookup getIndex)
  have nonnegative : ¬ (index : Int) < 0 := by omega
  simp [Mutable.getItem, found, Mutable.index, nonnegative, Nat.not_le.mpr inside, List.getElem?_eq_getElem inside, Except.map]

theorem pair_call (heap : Mutable.Heap) (state : Nat) (values : List Mutable.Value) (pairs : List Pair)
    (i j : Nat) (ordered : i < j) (upper : j < pairs.length) (canonical : ∀ pair ∈ pairs, Canonical pair)
    (env : HeapCalls.Env) (globals : Mutable.Globals) (found : heap[state]? = some (.tuple values))
    (rows : Heap.Rows heap values pairs) (getState : env "state" = some (.ref state))
    (getI : env "i" = some (.scalar (.int i))) (getJ : env "j" = some (.scalar (.int j))) :
    ∃ extra address operations,
      Evaluates program env pairCallExpr heap (.ref address) (heap ++ extra) globals ∧
      (heap ++ extra)[address]? = some (.list operations) ∧
      Heap.Rows (heap ++ extra) operations (choices (pairs[i]'(by omega)) pairs[j]) := by
  have sameLength := rows.length_eq
  have iInside : i < values.length := by omega
  have jInside : j < values.length := by omega
  have iPair : i < pairs.length := by omega
  have leftRow : Heap.Row heap values[i] pairs[i] := rows.get iInside iPair
  have rightRow : Heap.Row heap values[j] pairs[j] := rows.get jInside upper
  obtain ⟨left, leftValue, leftFound⟩ := leftRow
  obtain ⟨right, rightValue, rightFound⟩ := rightRow
  have leftRead := state_item heap state i values "i" env globals found iInside getState getI
  have rightRead := state_item heap state j values "j" env globals found jInside getState getJ
  rw [leftValue] at leftRead
  rw [rightValue] at rightRead
  obtain ⟨extra, returned, operations, listFound, operationRows⟩ :=
    pair_results.result pairs[i] pairs[j] left right heap globals
      (canonical pairs[i] (List.getElem_mem iPair)).positive (canonical pairs[j] (List.getElem_mem upper)).positive leftFound rightFound
  exact ⟨extra, pair_results.listAddress heap, operations,
    Evaluates.call (.cons leftRead (.cons rightRead .nil)) returned, listFound, operationRows⟩

/-- One increasing index pair executes the remainder comprehension and the entire operation-result loop. -/
theorem pair_body (heap : Mutable.Heap) (cell memo collector state : Nat)
    (values : List Mutable.Value) (pairs : List Pair) (i j : Nat) (ordered : i < j) (upper : j < pairs.length)
    (canonical : ∀ pair ∈ pairs, Canonical pair) (entries : List Memo.Entry) (collected : List Heap.SetEntry)
    (env : HeapCalls.Env) (globals : Mutable.Globals) (smaller : SmallerCalls pairs.length globals)
    (cellFound : heap[cell]? = some (.list [.ref memo])) (valid : MemoValid heap memo entries)
    (stateFound : heap[state]? = some (.tuple values)) (rows : Heap.Rows heap values pairs)
    (original : Heap.SetAt heap collector collected) (uncached : collector ∉ entries.map Memo.Entry.result)
    (getCell : env cellName = some (.ref cell)) (getState : env "state" = some (.ref state))
    (getI : env "i" = some (.scalar (.int i))) (getJ : env "j" = some (.scalar (.int j)))
    (getResult : env "results" = some (.ref collector)) :
    ∃ after finalEnv final finalEntries,
      BlockContinues program env pairBody heap finalEnv after globals ∧
      Heap.SetAt after collector final ∧
      (∀ pair, pair ∈ final.map Prod.fst ↔ pair ∈ collected.map Prod.fst ∨
        ∃ combined ∈ choices (pairs[i]'(by omega)) pairs[j], Expected (combined :: select (keepOperand i j) 0 pairs) pair) ∧
      MemoValid after memo finalEntries ∧ Memo.Origins entries heap.length finalEntries ∧
      collector ∉ finalEntries.map Memo.Entry.result ∧ PreservesOutside [memo, collector] heap after ∧
      (∀ name, name ≠ "rest" → name ≠ "value" → finalEnv name = env name) := by
  let remaining := select (keepOperand i j) 0 values
  let temporary : Mutable.Heap := [.list remaining]
  let prepared := heap ++ temporary
  let bound := setLocal env "rest" (.ref heap.length)
  have restEvaluated := remainder_expression heap state values i j env globals stateFound getState getI getJ
  have assigned : BlockContinues program env [.assign [.name "rest"] remainderExpr] heap bound prepared globals :=
    BlockContinues.assign restEvaluated
  have preparedState := HeapCollection.prefix_found heap temporary state _ stateFound
  obtain ⟨extra, operations, operationValues, called, operationsFound, operationRows⟩ :=
    pair_call prepared state values pairs i j ordered upper canonical bound globals preparedState (rows.append temporary)
      (by simpa [bound, setLocal] using getState) (by simpa [bound, setLocal] using getI) (by simpa [bound, setLocal] using getJ)
  have preparedRest : prepared[heap.length]? = some (.list remaining) := by simp [prepared, temporary]
  have finalRest := HeapCollection.prefix_found prepared extra heap.length _ preparedRest
  have restRows := (remainder_rows heap values pairs i j rows).append extra
  have liveCell := HeapCollection.prefix_found prepared extra cell _ (HeapCollection.prefix_found heap temporary cell _ cellFound)
  obtain ⟨after, finalEnv, final, finalEntries, loop, finalSet, members, finalMemo, origins, absent, frame, locals⟩ :=
    value_loop pairs i j ordered upper canonical globals smaller cell memo collector heap.length operations remaining operationValues
      (choices (pairs[i]'(by omega)) pairs[j]) operationValues 0 (prepared ++ extra) entries collected bound
      liveCell ((valid.append temporary).append extra) finalRest restRows operationsFound rfl operationRows
      (fun _ member => member) ((original.append temporary).append extra) uncached
      (by simpa [bound, setLocal, cellName] using getCell) (by simp [bound, setLocal]) (by simpa [bound, setLocal] using getResult)
  have traversed : BlockContinues program bound [.forIn (.name "value") pairCallExpr [.expr updateExpr] []]
      prepared finalEnv after globals := by
    apply BlockContinues.for_sequence (items := operationValues) (by intro expressions; simp [pairCallExpr]) called
      (by simp [Mutable.sequence, operationsFound])
    exact loop
  have preparationFrame : PreservesExcept memo heap (prepared ++ extra) :=
    (PreservesExcept.append memo heap temporary).trans (PreservesExcept.append memo prepared extra)
  refine ⟨after, finalEnv, final, finalEntries, assigned.append traversed, finalSet, members, finalMemo,
    (Memo.Origins.refl entries heap.length).trans origins preparationFrame.size, absent,
    (PreservesOutside.of_except preparationFrame (by simp)).trans frame, ?_⟩
  intro name notRest notValue
  rw [locals name notValue]
  simp [bound, setLocal, notRest]

end solvers.«93».all_results
