import solvers.«95.py.lean.d».WalkFrames
import lang.py.FormalSW.LoopCalls

/-! The actual first-repeat loop, including both early exits and cycle selection. -/
namespace solvers.«95».walk
open FormalSW.PyAst HeapCalls ProjectEulerSolutions
open BoundedCycles BoundedCycleWalk BoundedCycleReference
set_option Elab.async false

abbrev markCode : Stmt := .forIn (.name "v") (.name "path") processed.body []
abbrev outsideTest : Expr := .boolOp .or
  [.compare (.name "n") [(.eq, .constant (.int 0))], .compare (.name "n") [(.gt, .name "limit")]]
abbrev cycleCode : List Stmt := cycle.prepareCode ++
  [.ifThen cycle.condition cycle.replaceCode [], markCode, .breakLoop]
abbrev body : List Stmt :=
  [.ifThen outsideTest [markCode, .breakLoop] [],
   .ifThen (.subscript (.name "processed") (.name "n")) [markCode, .breakLoop] [],
   .ifThen (.membership (.name "n") (.name "idx") false) cycleCode []] ++ stepCode
abbrev loopCode : Stmt := .whileDo (.constant (.bool true)) body []
abbrev continuation : Stmt := Calls.loopContinuation body loopCode

variable {globals : Mutable.Globals}

theorem outside_test (heap : Mutable.Heap) (env : HeapCalls.Env) (a : Addresses)
    (limit current : Nat) (best : Nat × Nat) (captured : Captured env a limit current best) :
    Evaluates program env outsideTest heap (.scalar (.bool (decide (current = 0 ∨ limit < current)))) heap globals := by
  simp only [Bool.decide_or]
  apply solvers.«95».heap.boolean_pair .or
  · exact Evaluates.compare (Evaluates.lookup captured.getCurrent) Evaluates.constant
      (by simp [HeapCalls.compare, scalarCompare, evalCmpOp])
  · exact Evaluates.compare (Evaluates.lookup captured.getCurrent) (Evaluates.lookup captured.getLimit)
      (by simp [HeapCalls.compare, scalarCompare, evalCmpOp])

theorem mark_result (heap : Mutable.Heap) (env : HeapCalls.Env) (a : Addresses) (limit current : Nat)
    (f : Nat → Nat) (known : Finset Nat) (path : List Nat) (positions : NaturalDictionary.Entries)
    (best : Nat × Nat) (separate : a.Separated) (inside : ProjectEulerStatements.BoundedOrbit.Within limit path)
    (found : Found heap a limit f known path positions) (captured : Captured env a limit current best) :
    ∃ finalEnv, BlockContinues program env [markCode] heap finalEnv
      (heap.set a.processed (.list (processed.items limit (known ∪ path.toFinset)))) globals ∧
      Captured finalEnv a limit current best := by
  obtain ⟨finalEnv, executed, stable⟩ := processed.mark (globals := globals) limit a.processed a.path known path
    heap env separate.processedPath inside found.processed found.path captured.getProcessed captured.getPath
  refine ⟨finalEnv, executed, captured.frame ?_⟩
  intro name member
  apply stable
  simp only [List.mem_cons, List.not_mem_nil, or_false] at member
  rcases member with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl <;> decide

theorem cycle_score (f : Nat → Nat) (limit start current index : Nat) (path : List Nat)
    (visited : Prefix f limit start current path) (positive : 0 < current) (bounded : current ≤ limit)
    (distinct : path.Nodup) (selected : path[index]? = some current) :
    (path.drop index |>.length, least (path.drop index)) = score f limit current := by
  obtain ⟨_, members, length⟩ := repeated_cycle visited positive bounded distinct selected
  apply Prod.ext
  · exact length
  · simp only [least, members, dif_pos (orbit_nonempty f limit current), score]

/-- Each exit retains the divisor table and records exactly the completed path. -/
theorem result (heap : Mutable.Heap) (env : HeapCalls.Env) (a : Addresses) (limit start current : Nat)
    (f : Nat → Nat) (known : Finset Nat) (path : List Nat) (positions : NaturalDictionary.Entries)
    (best : Nat × Nat) (separate : a.Separated)
    (found : Found heap a limit f known path positions) (captured : Captured env a limit current best)
    (distinct : path.Nodup) (inside : ProjectEulerStatements.BoundedOrbit.Within limit path)
    (recorded : NaturalDictionary.Positions.Exact positions path) (visited : Prefix f limit start current path) :
    ∃ finished : Finished f limit known current path, ∃ finalEnv finalHeap finalPositions,
      BlockContinues program env [loopCode] heap finalEnv finalHeap globals ∧
      Found finalHeap a limit f (known ∪ finished.path.toFinset) finished.path finalPositions ∧
      Captured finalEnv a limit finished.current (winner best finished) := by
  have tested := outside_test (globals := globals) heap env a limit current best captured
  by_cases outside : current = 0 ∨ limit < current
  · obtain ⟨finalEnv, marked, finalCaptured⟩ := mark_result (globals := globals) heap env a limit current
      f known path positions best separate inside found captured
    refine ⟨⟨current, path, .exit, .exit outside, outside⟩, finalEnv, _, positions, ?_, found.marked separate, finalCaptured⟩
    apply BlockContinues.while_loop (flag := true) Evaluates.constant rfl
    apply BlockContinues.branch_next (flag := true) (by simpa only [outside, decide_true] using tested) rfl
    exact marked.append (BlockContinues.break_loop rfl .nil)
  · have positive : 0 < current := by omega
    have bounded : current ≤ limit := by omega
    have cachedTest := processed.read (globals := globals) limit current a.processed known bounded heap env
      found.processed captured.getProcessed captured.getCurrent
    by_cases cached : current ∈ known
    · obtain ⟨finalEnv, marked, finalCaptured⟩ := mark_result (globals := globals) heap env a limit current
        f known path positions best separate inside found captured
      refine ⟨⟨current, path, .cached, .cached positive bounded cached, positive, bounded, cached⟩,
        finalEnv, _, positions, ?_, found.marked separate, finalCaptured⟩
      apply BlockContinues.while_loop (flag := true) Evaluates.constant rfl
      apply BlockContinues.branch_next (flag := false) (by simpa only [outside, decide_false] using tested) rfl
      apply BlockContinues.branch_next (flag := true) (by simpa only [cached, decide_true] using cachedTest) rfl
      exact marked.append (BlockContinues.break_loop rfl .nil)
    · have memberTest := NaturalDictionary.membership_evaluates (program := program) (globals := globals)
        heap env "idx" "n" a.positions current positions found.positions captured.getPositions captured.getCurrent false
      cases selected : NaturalDictionary.lookup positions current with
      | some index =>
          have selectedPath := recorded.sound current index selected
          have before := (List.getElem?_eq_some_iff.mp selectedPath).1
          let suffix := path.drop index
          let allocated := heap ++ [.list (suffix.map number)]
          let prepared := cycle.ready env heap.length suffix
          have preparedFound : Found allocated a limit f known path positions := found.append _
          have preparedCaptured : Captured prepared a limit current best := captured.ready _ _
          have sliced := cycle.prepare (globals := globals) heap env a.path a.positions current index path positions
            before found.path found.positions captured.getPath captured.getPositions captured.getCurrent selected
          obtain ⟨selectedEnv, updated, length, minimum, stable⟩ := cycle.select_result (globals := globals)
            best (suffix.length, least suffix) allocated prepared (by simp [prepared, cycle.ready, setLocal])
            (by simp [prepared, cycle.ready, setLocal]) preparedCaptured.getLength preparedCaptured.getMinimum
          have scoreEq := cycle_score f limit start current index path visited positive bounded distinct selectedPath
          have chosenCaptured : Captured selectedEnv a limit current (choose best (score f limit current)) := by
            rw [← scoreEq]
            exact preparedCaptured.winner _ length minimum stable
          obtain ⟨finalEnv, marked, finalCaptured⟩ := mark_result (globals := globals) allocated selectedEnv a limit current
            f known path positions (choose best (score f limit current)) separate inside preparedFound chosenCaptured
          refine ⟨⟨current, path, .cycle index, .repeated positive bounded cached (List.mem_of_getElem? selectedPath),
              positive, bounded, cached, selectedPath⟩, finalEnv, _, positions, ?_, preparedFound.marked separate, finalCaptured⟩
          apply BlockContinues.while_loop (flag := true) Evaluates.constant rfl
          apply BlockContinues.branch_next (flag := false) (by simpa only [outside, decide_false] using tested) rfl
          apply BlockContinues.branch_next (flag := false) (by simpa only [cached, decide_false] using cachedTest) rfl
          apply BlockContinues.branch_next (flag := true) (by simpa only [selected] using memberTest) rfl
          exact sliced.append (updated.append (marked.append (BlockContinues.break_loop rfl .nil)))
      | none =>
          have fresh := (NaturalDictionary.Positions.missing_iff positions path recorded current).mp selected
          have advanced := iteration (globals := globals) heap env a limit current f known path positions best
            bounded separate found captured
          obtain ⟨finished, finalEnv, finalHeap, finalPositions, executed, finalFound, finalCaptured⟩ :=
            result (stepHeap heap a current path positions) (setLocal env "n" (number (f current))) a limit start (f current)
              f known (path ++ [current]) (NaturalDictionary.insert positions current path.length) best separate
              (found.advance separate current) (captured.advance (f current))
              (ProjectEulerStatements.BoundedOrbit.append_distinct _ _ distinct fresh)
              (ProjectEulerStatements.BoundedOrbit.append_within _ _ _ inside bounded)
              (NaturalDictionary.Positions.insert positions path recorded current fresh) (visited.advance positive bounded)
          refine ⟨⟨finished.current, finished.path, finished.stop,
              .next positive bounded cached fresh finished.trace, finished.selected⟩,
            finalEnv, finalHeap, finalPositions, ?_, finalFound, finalCaptured⟩
          apply BlockContinues.while_loop (flag := true) Evaluates.constant rfl
          apply BlockContinues.branch_next (flag := false) (by simpa only [outside, decide_false] using tested) rfl
          apply BlockContinues.branch_next (flag := false) (by simpa only [cached, decide_false] using cachedTest) rfl
          apply BlockContinues.branch_next (flag := false) (by simpa only [selected] using memberTest) rfl
          exact advanced.append (BlockContinues.loop_resume executed)
termination_by limit + 1 - path.length
decreasing_by
  have enough := ProjectEulerStatements.BoundedOrbit.room limit current path distinct inside bounded fresh
  simp only [List.length_append, List.length_singleton]
  omega

end solvers.«95».walk
