import «.formalsw».solvers.«29.py.ast»
import data.«project-euler-statements».data.py.project_euler_statements.«p29.py»
import lang.py.FormalSW.HeapCalls

namespace solvers.«29»
open FormalSW.PyAst FormalSW.PyAst.HeapCalls
open data.«project-euler-statements».data.py.project_euler_statements.p29
private abbrev program := Solvers.P29Ast.module
private abbrev num (n : Nat) : Mutable.Value := .scalar (.int n)
private abbrev body : List Stmt :=
  [.expr (.call (.attribute (.name "vals") "add") [.builtinCall .pow [.name "a", .name "b"]])]

private theorem inner (count start a : Nat) (stop : Int) (length : (stop - start).toNat = count)
    (heap : Mutable.Heap) (before : List Nat) (env : HeapCalls.Env)
    (getA : env "a" = some (num a)) (getSet : env "vals" = some (.ref heap.length)) :
    ∃ finalEnv, BlockContinues program env [.rangeStepLoop "b" start stop 1 body []]
      (heap ++ [.set (keys before)]) finalEnv
      (heap ++ [.set (keys (((List.range' start count).map (a ^ ·)).foldl insertUnique before))]) ∧
      finalEnv "a" = env "a" ∧ finalEnv "max_b" = env "max_b" ∧ finalEnv "vals" = env "vals" := by
  induction count generalizing start before env with
  | zero =>
      have finished : stop ≤ start := by omega
      refine ⟨env, ?_, rfl, rfl, rfl⟩
      apply BlockContinues.range_done
      · simp [Calls.rangeContinues]; omega
      · exact .nil
  | succ count ih =>
      have continues : (start : Int) < stop := by omega
      let bound := setLocal env "b" (num start)
      have getA' : bound "a" = some (num a) := by simpa [bound, setLocal] using getA
      have getSet' : bound "vals" = some (.ref heap.length) := by simpa [bound, setLocal] using getSet
      have found : (heap ++ [.set (keys before)])[heap.length]? = some (.set (keys before)) := by simp
      have power : Evaluates program bound (.builtinCall .pow [.name "a", .name "b"])
          (heap ++ [.set (keys before)]) (num (a ^ start)) (heap ++ [.set (keys before)]) := by
        apply Evaluates.builtin (.cons (Evaluates.lookup getA') (.cons (Evaluates.lookup rfl) .nil))
        simp [primitive, scalarResult, scalarBinary, num, show ¬ (start : Int) < 0 by omega]
      have added : BlockContinues program bound body (heap ++ [.set (keys before)]) bound
          (heap ++ [.set (keys (insertUnique before (a ^ start)))]) := by
        apply BlockContinues.discard (value := .scalar .none)
        have keyEq : ScalarSet.insert (keys before) (.int ((a : Int) ^ start)) =
            keys (insertUnique before (a ^ start)) := by
          simpa only [Nat.cast_pow] using insert_keys before (a ^ start)
        simpa [keyEq] using Evaluates.add (Evaluates.lookup getSet') found power found
      obtain ⟨finalEnv, rest, finalA, finalB, finalSet⟩ :=
        ih (start + 1) (by omega) (insertUnique before (a ^ start)) bound getA' getSet'
      refine ⟨finalEnv, ?_, by simpa [bound, setLocal] using finalA,
        by simpa [bound, setLocal] using finalB, by simpa [bound, setLocal] using finalSet⟩
      apply BlockContinues.range_step (by simp [Calls.rangeContinues, continues])
      simpa [Calls.loopContinuation, Calls.loopControl, Calls.loopControlStmt, body,
        List.range'_succ, Nat.cast_add, Nat.cast_one, bound] using added.append rest

private abbrev outerBody : List Stmt :=
  [.forIn (.name "b") (.call (.name "range")
    [.constant (.int 2), .binOp (.name "max_b") .add (.constant (.int 1))]) body []]

private theorem outer (count start : Nat) (stop maxB : Int) (length : (stop - start).toNat = count)
    (heap : Mutable.Heap) (before : List Nat) (env : HeapCalls.Env)
    (getB : env "max_b" = some (.scalar (.int maxB))) (getSet : env "vals" = some (.ref heap.length)) :
    ∃ finalEnv, BlockContinues program env [.rangeStepLoop "a" start stop 1 outerBody []]
      (heap ++ [.set (keys before)]) finalEnv
      (heap ++ [.set (keys (((List.range' start count).flatMap
        (fun a => row a (maxB - 1).toNat)).foldl insertUnique before))]) ∧
      finalEnv "max_b" = env "max_b" ∧ finalEnv "vals" = env "vals" := by
  induction count generalizing start before env with
  | zero =>
      have finished : stop ≤ start := by omega
      refine ⟨env, ?_, rfl, rfl⟩
      apply BlockContinues.range_done
      · simp [Calls.rangeContinues]; omega
      · exact .nil
  | succ count ih =>
      have continues : (start : Int) < stop := by omega
      let bound := setLocal env "a" (num start)
      let afterRow := (row start (maxB - 1).toNat).foldl insertUnique before
      have getB' : bound "max_b" = some (.scalar (.int maxB)) := by simpa [bound, setLocal] using getB
      have getSet' : bound "vals" = some (.ref heap.length) := by simpa [bound, setLocal] using getSet
      obtain ⟨nextEnv, innerRun, _, nextB, nextSet⟩ :=
        inner (maxB - 1).toNat 2 start (maxB + 1) (by congr 1; omega) heap before bound rfl getSet'
      have rowRun : BlockContinues program bound outerBody (heap ++ [.set (keys before)]) nextEnv
          (heap ++ [.set (keys afterRow)]) := by
        apply BlockContinues.for_range (values := [.scalar (.int 2), .scalar (.int (maxB + 1))])
          (middle := heap ++ [.set (keys before)])
        · exact .cons Evaluates.constant
            (.cons (Evaluates.binary (Evaluates.lookup getB') Evaluates.constant rfl) .nil)
        · rfl
        · exact innerRun
      obtain ⟨finalEnv, rest, finalB, finalSet⟩ := ih (start + 1) (by omega) afterRow nextEnv
        (nextB.trans getB') (nextSet.trans getSet')
      refine ⟨finalEnv, ?_, finalB.trans (nextB.trans (by simp [bound, setLocal])),
        finalSet.trans (nextSet.trans (by simp [bound, setLocal]))⟩
      apply BlockContinues.range_step (by simp [Calls.rangeContinues, continues])
      simpa [Calls.loopContinuation, Calls.loopControl, Calls.loopControlStmt, outerBody, body,
        List.range'_succ, List.foldl_append, Nat.cast_add, Nat.cast_one, bound, afterRow] using rowRun.append rest

theorem count_distinct_powers.result (a b : Int) (heap : Mutable.Heap) :
    ReturnsValue program "count_distinct_powers" [.scalar (.int a), .scalar (.int b)] heap
      (num (values a b).eraseDups.length) (heap ++ [.set (keys (values a b).eraseDups)]) := by
  apply BlockReturns.function rfl rfl
  apply BlockContinues.returns (BlockContinues.annAssign
    (value := .ref heap.length) (finalHeap := heap ++ [.set []]) (Evaluates.builtin .nil rfl))
  let env := setLocal (HeapCalls.argumentEnv
    [{ name := "max_a", annotation := some (.name "int") }, { name := "max_b", annotation := some (.name "int") }]
      [.scalar (.int a), .scalar (.int b)]) "vals" (.ref heap.length)
  obtain ⟨finalEnv, loops, _, finalSet⟩ := outer (a - 1).toNat 2 (a + 1) b
    (by congr 1; omega) heap [] env rfl rfl
  have collected : BlockContinues program env [Stmt.forIn (.name "a") (.call (.name "range")
      [.constant (.int 2), .binOp (.name "max_a") .add (.constant (.int 1))]) outerBody []]
      (heap ++ [.set []]) finalEnv (heap ++ [.set (keys (values a b).eraseDups)]) := by
    apply BlockContinues.for_range (values := [.scalar (.int 2), .scalar (.int (a + 1))])
      (middle := heap ++ [.set []])
    · exact .cons Evaluates.constant
        (.cons (Evaluates.binary (Evaluates.lookup rfl) Evaluates.constant rfl) .nil)
    · rfl
    · have reduced : (values a b).foldl insertUnique [] = (values a b).eraseDups := by
        simpa using fold_insert (values a b) []
      change BlockContinues program env [.rangeStepLoop "a" 2 (a + 1) 1 outerBody []]
        (heap ++ [.set (keys [])]) finalEnv (heap ++ [.set (keys ((values a b).foldl insertUnique []))]) at loops
      rw [reduced] at loops
      exact loops
  apply collected.returns
  apply BlockReturns.return_value
  apply Evaluates.builtin (.cons (Evaluates.lookup finalSet) .nil)
  simp [primitive, Mutable.length, keys, num]

theorem count_distinct_powers.«Equiv(naive)» (a b : Int) :
    (HeapCalls.scalarFunction program "count_distinct_powers" [.int a, .int b]).Equiv
      (HeapCalls.scalarFunction P29Ast.module "naive" [.int a, .int b]) := by
  have optimized := HeapCalls.ReturnsValue.scalar_finishes (values := [.int a, .int b]) (count_distinct_powers.result a b [])
  have reference := HeapCalls.ReturnsValue.scalar_finishes (values := [.int a, .int b]) (data.«project-euler-statements».data.py.project_euler_statements.p29.naive.result a b [])
  intro outcome
  constructor
  · intro finished
    rw [finished.unique optimized]
    exact reference
  · intro finished
    rw [finished.unique reference]
    exact optimized

theorem count_distinct_powers.«Supported» :
    (∀ a b : Int, (HeapCalls.scalarFunction program "count_distinct_powers" [.int a, .int b]).Supported) ∧
    (∀ a b : Int, (HeapCalls.scalarFunction P29Ast.module "naive" [.int a, .int b]).Supported) := by
  constructor
  · intro a b
    exact (count_distinct_powers.result a b []).scalar_returns.supported
  · intro a b
    exact (data.«project-euler-statements».data.py.project_euler_statements.p29.naive.«Returns» a b).supported

end solvers.«29»
