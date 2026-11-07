import solvers.«95.py.lean.d».Walk

/-! The outer source scan retains a sound processed cache and the best cycle found so far. -/
namespace solvers.«95».search
open FormalSW.PyAst HeapCalls ProjectEulerSolutions
open BoundedCycles BoundedCycleWalk
set_option Elab.async false

abbrev program := Solvers.P95Ast.module
abbrev number (n : Nat) : Mutable.Value := .scalar (.int n)

structure Locals (env : HeapCalls.Env) (limit table processed : Nat) (best : Nat × Nat) : Prop where
  getLimit : env "limit" = some (number limit)
  getTable : env "s" = some (.ref table)
  getProcessed : env "processed" = some (.ref processed)
  getLength : env "best_len" = some (number best.1)
  getMinimum : env "best_min" = some (number best.2)

structure Tables (heap : Mutable.Heap) (limit table processedAddress : Nat) (f : Nat → Nat)
    (known : Finset Nat) : Prop where
  table : heap[table]? = some (.list (Sieve.table (limit + 1) f))
  processed : heap[processedAddress]? = some (.list (processed.items limit known))

structure Invariant (f : Nat → Nat) (limit next : Nat) (known : Finset Nat) (best : Nat × Nat) : Prop where
  attained : Attained f limit best
  cached : CacheValid f limit best known
  covered : ∀ n, n < next → Safe f limit best n

abbrev freshCode : List Stmt :=
  [.annAssign (.name "path") (.subscript (.name "List") (.name "int")) (some (.list [])),
   .annAssign (.name "idx") (.subscript (.name "Dict") (.tuple [.name "int", .name "int"])) (some (.dict [])),
   .assign [.name "n"] (.name "start")]
abbrev body : List Stmt :=
  [.ifThen (.subscript (.name "processed") (.name "start")) [.continueLoop] []] ++ freshCode ++ [walk.loopCode]

variable {globals : Mutable.Globals}

theorem prepare (heap : Mutable.Heap) (env : HeapCalls.Env) (limit table processedAddress start : Nat)
    (f : Nat → Nat) (known : Finset Nat) (best : Nat × Nat) (different : table ≠ processedAddress)
    (tables : Tables heap limit table processedAddress f known) (locals : Locals env limit table processedAddress best)
    (getStart : env "start" = some (number start)) :
    ∃ finalEnv, let addresses : walk.Addresses := ⟨table, processedAddress, heap.length, heap.length + 1⟩
      BlockContinues program env freshCode heap finalEnv (heap ++ [.list [], .dict []]) globals ∧
      addresses.Separated ∧ walk.Found (heap ++ [.list [], .dict []]) addresses limit f known [] [] ∧
      walk.Captured finalEnv addresses limit start best := by
  let pathEnv := setLocal env "path" (.ref heap.length)
  let indexEnv := setLocal pathEnv "idx" (.ref (heap.length + 1))
  let ready := setLocal indexEnv "n" (number start)
  refine ⟨ready, ?_, ?_, ?_, ?_⟩
  · have first := BlockContinues.annAssign (annotation := .subscript (.name "List") (.name "int"))
      (globals := globals) (module := program) (env := env)
      (name := "path") (Evaluates.list (expressions := []) (heap := heap) .nil)
    have second := BlockContinues.annAssign (annotation := .subscript (.name "Dict") (.tuple [.name "int", .name "int"]))
      (globals := globals) (module := program) (env := pathEnv)
      (name := "idx") (Evaluates.dictionary (heap := heap ++ [.list []]) .nil)
    simp only [List.length_append, List.length_singleton, List.append_assoc, List.singleton_append] at second
    have third : BlockContinues program indexEnv [.assign [.name "n"] (.name "start")]
        (heap ++ [.list [], .dict []]) ready (heap ++ [.list [], .dict []]) globals :=
      BlockContinues.assign (Evaluates.lookup (by simpa [indexEnv, pathEnv, setLocal] using getStart))
    simpa only [List.length_append, List.length_singleton, List.append_assoc, List.singleton_append,
      freshCode, ready, indexEnv, pathEnv] using first.append (second.append third)
  · have tableBound := (List.getElem?_eq_some_iff.mp tables.table).1
    have processedBound := (List.getElem?_eq_some_iff.mp tables.processed).1
    exact ⟨different, by dsimp; omega, by dsimp; omega, by dsimp; omega, by dsimp; omega, by dsimp; omega⟩
  · constructor
    · exact (List.getElem?_append_left (List.getElem?_eq_some_iff.mp tables.table).1).trans tables.table
    · exact (List.getElem?_append_left (List.getElem?_eq_some_iff.mp tables.processed).1).trans tables.processed
    · simp
    · simp [NaturalDictionary.Found, NaturalDictionary.encode]
  · constructor
    · simpa [ready, indexEnv, pathEnv, setLocal] using locals.getLimit
    · simpa [ready, indexEnv, pathEnv, setLocal] using locals.getTable
    · simpa [ready, indexEnv, pathEnv, setLocal] using locals.getProcessed
    · simp [ready, indexEnv, pathEnv, setLocal]
    · simp [ready, indexEnv, pathEnv, setLocal]
    · simp [ready, setLocal]
    · simpa [ready, indexEnv, pathEnv, setLocal] using locals.getLength
    · simpa [ready, indexEnv, pathEnv, setLocal] using locals.getMinimum

theorem range_result (heap : Mutable.Heap) (env : HeapCalls.Env) (limit table processedAddress next : Nat)
    (f : Nat → Nat) (known : Finset Nat) (best : Nat × Nat) (different : table ≠ processedAddress)
    (tables : Tables heap limit table processedAddress f known) (locals : Locals env limit table processedAddress best)
    (valid : Invariant f limit next known best) :
    ∃ finalHeap finalEnv finalKnown finalBest,
      BlockContinues program env [.rangeStepLoop "start" next (limit + 1) 1 body []] heap finalEnv finalHeap globals ∧
      Tables finalHeap limit table processedAddress f finalKnown ∧ Locals finalEnv limit table processedAddress finalBest ∧
      Optimal f limit finalBest := by
  by_cases active : next < limit + 1
  · let entered := setLocal env "start" (number next)
    have enteredLocals : Locals entered limit table processedAddress best := by
      exact ⟨by simpa [entered, setLocal] using locals.getLimit, by simpa [entered, setLocal] using locals.getTable,
        by simpa [entered, setLocal] using locals.getProcessed, by simpa [entered, setLocal] using locals.getLength,
        by simpa [entered, setLocal] using locals.getMinimum⟩
    have getStart : entered "start" = some (number next) := by simp [entered, setLocal]
    have tested : Evaluates program entered (.subscript (.name "processed") (.name "start")) heap
        (.scalar (.bool (decide (next ∈ known)))) heap globals := by
      apply Evaluates.subscript (by intros; simp) (Evaluates.lookup enteredLocals.getProcessed)
        (Evaluates.lookup getStart)
      have selected := processed.items_lookup limit next known (by omega)
      simpa only [(List.getElem?_eq_some_iff.mp selected).2] using
        Mutable.getItem_list heap processedAddress next _ tables.processed (by simp; omega)
    by_cases cached : next ∈ known
    · have nextValid : Invariant f limit (next + 1) known best := ⟨valid.attained, valid.cached, by
        intro n before
        by_cases same : n = next
        · subst n; exact valid.cached next cached
        · exact valid.covered n (by omega)⟩
      obtain ⟨finalHeap, finalEnv, finalKnown, finalBest, executed, finalTables, finalLocals, optimal⟩ :=
        range_result heap entered limit table processedAddress (next + 1) f known best different tables enteredLocals nextValid
      refine ⟨finalHeap, finalEnv, finalKnown, finalBest, ?_, finalTables, finalLocals, optimal⟩
      apply BlockContinues.range_step (by simp [Calls.rangeContinues]; omega)
      apply BlockContinues.branch_next (flag := true) (by simpa only [cached, decide_true] using tested) rfl
      simpa only [Nat.cast_add, Nat.cast_one] using BlockContinues.continue_loop rfl executed
    · obtain ⟨ready, prepared, separate, walkFound, walkCaptured⟩ := prepare (globals := globals) heap entered
        limit table processedAddress next f known best different tables enteredLocals (by simp [entered, setLocal])
      obtain ⟨finished, walkedEnv, walkedHeap, positions, walked, walkedFound, walkedCaptured⟩ :=
        walk.result (globals := globals) (heap ++ [.list [], .dict []]) ready
          ⟨table, processedAddress, heap.length, heap.length + 1⟩ limit next next f known [] [] best
          separate walkFound walkCaptured (by simp) (by simp [ProjectEulerStatements.BoundedOrbit.Within])
          NaturalDictionary.Positions.empty (Prefix.initial f limit next (by omega))
      have correct := finished_correct f limit known next (by omega) best valid.attained valid.cached finished
      have nextValid : Invariant f limit (next + 1) (known ∪ finished.path.toFinset) (winner best finished) :=
        ⟨correct.1, correct.2.2.2, by
          intro n before
          by_cases same : n = next
          · subst n; exact correct.2.2.1
          · exact (valid.covered n (by omega)).mono correct.2.1⟩
      have nextTables : Tables walkedHeap limit table processedAddress f (known ∪ finished.path.toFinset) :=
        ⟨walkedFound.table, walkedFound.processed⟩
      have nextLocals : Locals walkedEnv limit table processedAddress (winner best finished) :=
        ⟨walkedCaptured.getLimit, walkedCaptured.getTable, walkedCaptured.getProcessed,
          walkedCaptured.getLength, walkedCaptured.getMinimum⟩
      obtain ⟨finalHeap, finalEnv, finalKnown, finalBest, executed, finalTables, finalLocals, optimal⟩ :=
        range_result walkedHeap walkedEnv limit table processedAddress (next + 1) f
          (known ∪ finished.path.toFinset) (winner best finished) different nextTables nextLocals nextValid
      refine ⟨finalHeap, finalEnv, finalKnown, finalBest, ?_, finalTables, finalLocals, optimal⟩
      apply BlockContinues.range_step (by simp [Calls.rangeContinues]; omega)
      apply BlockContinues.branch_next (flag := false) (by simpa only [cached, decide_false] using tested) rfl
      have resumed := BlockContinues.loop_resume executed
      simpa only [Nat.cast_add, Nat.cast_one, Calls.loopContinuation, Calls.loopControl, Calls.loopControlStmt,
        body, freshCode, Bool.false_eq_true, if_false, List.append_assoc] using prepared.append (walked.append resumed)
  · refine ⟨heap, env, known, best, BlockContinues.range_done (by simp [Calls.rangeContinues]; omega) .nil,
      tables, locals, valid.attained, ?_⟩
    intro n cycle
    exact valid.covered n (by have bounded := cyclic_bounded cycle; omega) n (start_mem f limit n) cycle
termination_by limit + 1 - next
decreasing_by all_goals omega

end solvers.«95».search
