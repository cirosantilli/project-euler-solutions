import solvers.«96.py.lean.d».UnitLists

namespace solvers.«96».build_peers_units
open FormalSW.PyAst HeapCalls ProjectEulerSolutions ProjectEulerSolutions.Sudoku reduce_cands
set_option Elab.async false
noncomputable section

abbrev lower (name : String) : Expr := .binOp (.name name) .mult (.constant (.int 3))
abbrev upper (name : String) : Expr := .binOp (lower name) .add (.constant (.int 3))
abbrev appendCell : List Stmt := [.expr (.call (.attribute (.name "box") "append") [cellExpression])]
abbrev boxRow : Stmt := .forIn (.name "c") (.call (.name "range") [lower "bc", upper "bc"]) appendCell []
abbrev boxRows : Stmt := .forIn (.name "r") (.call (.name "range") [lower "br", upper "br"]) [boxRow] []
abbrev boxBody : List Stmt := [.assign [.name "box"] (.list []), boxRows,
  .expr (.call (.attribute (.name "units") "append") [.name "box"])]

theorem box_source : definition.2[3]? = some
    (.forIn (.name "br") (.call (.name "range") [.constant (.int 3)])
      [.forIn (.name "bc") (.call (.name "range") [.constant (.int 3)]) boxBody []] []) := rfl

theorem bounds (name : String) (coordinate : Nat) (env : HeapCalls.Env) (heap : Mutable.Heap) (globals : Mutable.Globals)
    (getCoordinate : env name = some (number coordinate)) :
    Evaluates.Arguments program env [lower name, upper name] heap [number (3 * coordinate), number (3 * coordinate + 3)] heap globals := by
  have multiplied : Evaluates program env (lower name) heap (number (3 * coordinate)) heap globals := by
    apply Evaluates.binary (Evaluates.lookup getCoordinate) Evaluates.constant
    simp [Mutable.binary, scalarBinary, number, Int.mul_comm]
  refine .cons multiplied (.cons ?_ .nil)
  apply Evaluates.binary multiplied Evaluates.constant
  simp [Mutable.binary, scalarBinary, number]

def rowValues (r c : Nat) : List Mutable.Value :=
  (List.finRange 3).map fun offset => number (r * 9 + (3 * c + offset.val))

theorem box_row (r c : Nat) (env : HeapCalls.Env) (base : Mutable.Heap) (output : List Mutable.Value)
    (globals : Mutable.Globals) (getBox : env "box" = some (.ref base.length))
    (getRow : env "r" = some (number r)) (getColumn : env "bc" = some (number c)) :
    ∃ finalEnv, BlockContinues program env [boxRow] (HeapCollection.working base output []) finalEnv
        (HeapCollection.working base (output ++ rowValues r c) []) globals ∧
      (∀ name, name ≠ "c" → finalEnv name = env name) := by
  let storage := fun values => HeapCollection.working base values []
  let invariant := fun scope : HeapCalls.Env => ∀ name, name ≠ "c" → scope name = env name
  let update := fun values (offset : Fin 3) => values ++ [number (r * 9 + (3 * c + offset.val))]
  have callback (values : List Mutable.Value) (scope : HeapCalls.Env) (offset : Fin 3) (ready : invariant scope)
      (getOffset : scope "c" = some (.scalar (.int (3 * c + 1 * offset.val)))) :
      ∃ afterEnv, BlockContinues program scope appendCell (storage values) afterEnv (storage (update values offset)) globals ∧
        invariant afterEnv := by
    have getBox' := (ready "box" (by decide)).trans getBox
    have getRow' := (ready "r" (by decide)).trans getRow
    have evaluated := linear r (3 * c + offset.val) scope (storage values) globals getRow' (by simpa using getOffset)
    refine ⟨scope, ?_, ready⟩
    have appended := Evaluates.append (Evaluates.lookup getBox') (HeapCollection.found base values [])
      evaluated (HeapCollection.found base values [])
    simpa only [storage, update, HeapCollection.store] using BlockContinues.discard appended
  obtain ⟨finalEnv, executed, preserved⟩ := HeapFiniteRange.loop program "c" appendCell rfl 3 (3 * c) 1 (by decide)
    storage update invariant globals
    (by intro scope offset ready name different; simpa [setLocal, different] using ready name different)
    callback (List.finRange 3) 0 rfl (by omega) output env (by intro _ _; rfl)
  refine ⟨finalEnv, ?_, preserved⟩
  apply BlockContinues.for_range (bounds "bc" c env (storage output) globals getColumn) rfl
  simpa [update, storage, rowValues, List.finRange_succ, List.finRange_zero, List.append_assoc] using executed

/-- The original row and column loops create each nine-cell box in row-major order. -/
theorem collect_box (r c : Fin 3) (env : HeapCalls.Env) (heap : Mutable.Heap) (globals : Mutable.Globals)
    (getRow : env "br" = some (number r.val)) (getColumn : env "bc" = some (number c.val)) :
    ∃ finalEnv, BlockContinues program env (boxBody.take 2) heap finalEnv
        (heap ++ [.list (peerLoop.encoded ((List.finRange 9).map (boxCell r c)))]) globals ∧
      finalEnv "box" = some (.ref heap.length) ∧
      (∀ name, name ∉ ["box", "r", "c"] → finalEnv name = env name) := by
  let ready := setLocal env "box" (.ref heap.length)
  let storage := fun values => HeapCollection.working heap values []
  let invariant := fun scope : HeapCalls.Env => scope "box" = some (.ref heap.length) ∧
    ∀ name, name ∉ ["box", "r", "c"] → scope name = env name
  let update := fun values (offset : Fin 3) => values ++ rowValues (3 * r.val + offset.val) c.val
  have callback (values : List Mutable.Value) (scope : HeapCalls.Env) (offset : Fin 3) (valid : invariant scope)
      (getOffset : scope "r" = some (.scalar (.int (3 * r.val + 1 * offset.val)))) :
      ∃ afterEnv, BlockContinues program scope [boxRow] (storage values) afterEnv (storage (update values offset)) globals ∧
        invariant afterEnv := by
    obtain ⟨afterEnv, executed, preserved⟩ := box_row (3 * r.val + offset.val) c.val scope heap values globals valid.1
      (by simpa using getOffset) ((valid.2 "bc" (by decide)).trans getColumn)
    exact ⟨afterEnv, executed, (preserved "box" (by decide)).trans valid.1,
      fun name outside => (preserved name (fun same => outside (by simp [same]))).trans
        (valid.2 name outside)⟩
  obtain ⟨finalEnv, executed, valid⟩ := HeapFiniteRange.loop program "r" [boxRow] rfl 3 (3 * r.val) 1 (by decide)
    storage update invariant globals
    (by
      intro scope offset valid
      refine ⟨by simpa [setLocal] using valid.1, ?_⟩
      intro name outside
      have different : name ≠ "r" := by simp only [List.mem_cons, List.not_mem_nil, or_false, not_or] at outside; exact outside.2.1
      simpa [setLocal, different] using valid.2 name outside)
    callback (List.finRange 3) 0 rfl (by omega) [] ready
    ⟨by simp [ready, setLocal], by intro name outside; simp only [List.mem_cons, List.not_mem_nil, or_false, not_or] at outside; simp [ready, setLocal, outside.1]⟩
  have collected : BlockContinues program ready [boxRows] (storage []) finalEnv
      (storage ((List.finRange 3).foldl update [])) globals := by
    apply BlockContinues.for_range (bounds "br" r.val ready (storage []) globals (by simpa [ready, setLocal] using getRow)) rfl
    simpa only [Nat.mul_zero, Nat.add_zero, Nat.one_mul] using executed
  refine ⟨finalEnv, ?_, valid.1, valid.2⟩
  have assigned : BlockContinues program env [.assign [.name "box"] (.list [])] heap ready (storage []) globals :=
    BlockContinues.assign (Evaluates.list .nil)
  have cells : (List.finRange 3).foldl update [] = peerLoop.encoded ((List.finRange 9).map (boxCell r c)) := by
    simp [update, rowValues, peerLoop.encoded, boxCell, Encoding.index, List.finRange_succ, List.finRange_zero]
  simpa only [cells] using assigned.append collected

end
end solvers.«96».build_peers_units
