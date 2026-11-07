import solvers.«18.py.lean.d».RowRuntime

namespace solvers.«18».runtimeCost
open FormalSW.PyAst HeapCalls HeapCalls.Runtime
open HeapCalls.Runtime.NestedIntList (encode)
private abbrev program := Solvers.P18Ast.module

abbrev TriangleRows := List (Nat × List Int)
def rowWork (rows : TriangleRows) : Nat := (rows.map (fun pair => pair.2.length+1)).sum

def triangleBody : List Stmt :=
  match executableBody ((findFunctionIn program.body "solve").getD ([], [])).2 with
  | [_, .forIn _ _ statements _, _] => statements
  | _ => []
def finalReturn : Stmt :=
  match executableBody ((findFunctionIn program.body "solve").getD ([], [])).2 with
  | [_, _, statement] => statement
  | _ => .return (.constant .none)

private theorem prefix_read (heap extra : Heap) (address : Nat) (object : Mutable.Object)
    (found : heap[address]? = some object) : (heap ++ extra)[address]? = some object := by
  rw [List.getElem?_append_left (List.getElem?_eq_some_iff.mp found).choose]
  exact found

/-- Each row is read once and produces one private dynamic-programming row.
The input heap may contain shared row references; it remains an unchanged prefix. -/
theorem triangle_loop (remaining : Nat) (heap : Heap) (env : HeapCalls.Env)
    (triangleAddress belowAddress : Nat) (rows : TriangleRows) (below : List Int)
    (getTriangle : heap[triangleAddress]? = some (.list (rows.map (fun pair => .ref pair.1))))
    (getRows : ∀ pair ∈ rows, heap[pair.1]? = some (.list (encode pair.2)))
    (shape : ∀ (i : Nat) (valid : i < rows.length), rows[i].2.length = i+1)
    (bounded : remaining ≤ rows.length)
    (getBelow : heap[belowAddress]? = some (.list (encode below))) (size : below.length = remaining+1)
    (localTriangle : env "triangle" = some (.ref triangleAddress))
    (localBelow : env "dp" = some (.ref belowAddress)) :
    ∃ (extra : Heap) (answer : Int), BlockReturnsIn .word program env
      [.rangeStepLoop "r" ((remaining : Int)-1) (-1) (-1) triangleBody [], finalReturn]
      heap (100*rowWork (rows.take remaining)+15) (.scalar (.int answer)) (heap ++ extra) := by
  induction remaining generalizing heap env belowAddress below with
  | zero =>
      refine ⟨[], below[0]'(by omega), ?_⟩
      have read : EvaluatesIn .word program env (.subscript (.name "dp") (.constant (.int 0))) heap 10
          (.scalar (.int (below[0]'(by omega)))) heap := by
        have run := EvaluatesIn.list_subscript (arithmetic := .word) (module := program) (index := .constant (.int 0))
          (by intros a b c h; cases h) (EvaluatesIn.lookup localBelow) EvaluatesIn.constant getBelow
          (Mutable.getItem_list heap belowAddress 0 (encode below) getBelow (by simpa using (show 0 < below.length by omega)))
        simpa using run
      have done := BlockContinuesIn.range_done (globals := []) (module := program) (env := env) (name := "r")
        (next := -1) (stop := -1) (stride := -1) (body := triangleBody) (otherwise := []) (heap := heap)
        rfl BlockContinuesIn.nil
      simpa [rowWork, finalReturn] using done.returns (BlockReturnsIn.return_value read)
  | succ remaining ih =>
      have valid : remaining < rows.length := by omega
      let pair := rows[remaining]
      have rowSize : pair.2.length = remaining+1 := shape remaining valid
      let indexed := setLocal env "r" (.scalar (.int remaining))
      let rowEnv := setLocal indexed "row" (.ref pair.1)
      have read : EvaluatesIn .word program indexed (.subscript (.name "triangle") (.name "r")) heap 10
          (.ref pair.1) heap := by
        have run := EvaluatesIn.list_subscript (arithmetic := .word) (module := program) (index := .name "r")
          (by intros a b c h; cases h)
          (EvaluatesIn.lookup (name := "triangle") (by simpa [indexed, setLocal] using localTriangle))
          (EvaluatesIn.lookup (env := indexed) (name := "r") rfl) getTriangle
          (Mutable.getItem_list heap triangleAddress remaining (rows.map (fun pair => .ref pair.1)) getTriangle (by simpa using valid))
        simpa using run
      obtain ⟨newRow, newSize, calculated⟩ := row_cost heap rowEnv pair.1 belowAddress pair.2 below
        (getRows pair (List.getElem_mem valid)) getBelow (by omega) rfl
        (by simpa [rowEnv, indexed, setLocal] using localBelow)
      let extraRow := [Mutable.Object.list (encode newRow)]
      let assigned := setLocal rowEnv "dp" (.ref heap.length)
      obtain ⟨extra, answer, tail⟩ := ih (heap ++ extraRow) assigned heap.length newRow
        (prefix_read heap extraRow _ _ getTriangle)
        (fun item member => prefix_read heap extraRow _ _ (getRows item member))
        (by omega) (by simp [extraRow]) (by omega)
        (by simpa [assigned, rowEnv, indexed, setLocal] using localTriangle) rfl
      have first := (BlockContinuesIn.assign (name := "row") read).append
        (BlockContinuesIn.assign (name := "dp") calculated)
      have step := BlockReturnsIn.range_step (env := env) (name := "r") (next := remaining)
        (stop := -1) (stride := -1) (body := triangleBody) (otherwise := []) (following := [finalReturn])
        (by simp [Calls.rangeContinues]; omega)
        (by simpa [triangleBody, Calls.loopContinuation, show (remaining : Int)-1 = (remaining : Int)+(-1) by omega] using first.returns tail)
      have width : rowWork (rows.take (remaining+1)) = rowWork (rows.take remaining) + pair.2.length+1 := by
        rw [List.take_succ_eq_append_getElem valid]
        simp only [rowWork, List.map_append, List.map_cons, List.map_nil, List.sum_append,
          List.sum_cons, List.sum_nil, Nat.add_zero, Nat.add_assoc]
        rfl
      refine ⟨extraRow ++ extra, answer, ?_⟩
      rw [show ((remaining+1 : Nat) : Int)-1 = (remaining : Int) by omega]
      apply BlockReturnsIn.mono
      · simpa only [List.append_assoc] using step
      · rw [width]
        omega

end solvers.«18».runtimeCost
