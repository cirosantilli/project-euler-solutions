import solvers.«93.py.lean.d».Winner

namespace solvers.«93».solve
open FormalSW.PyAst HeapCalls ProjectEulerSolutions.RationalPair
open ProjectEulerSolutions.HeapCollection
open data.«project-euler-statements».data.py.project_euler_statements.p93.consecutive_for_digits (permutationSources)
open data.«project-euler-statements».data.py.project_euler_statements.p93.naive (bestScore nextBest scan operands candidates answer)
set_option Elab.async false


abbrev endpoint : Expr := .binOp (.name "max_digit") .add (.constant (.int 1))
abbrev poolExpression : Expr := .listComp (.name "$formalsw.range.item")
  [("$formalsw.range.item", .call (.name "range") [endpoint], [])]

theorem source_body : executableBody ((findFunctionIn program.body "solve").getD ([], [])).2 =
    [.assign [.name "best_length"] (.unaryOp .uSub (.constant (.int 1))),
     .assign [.name "best_digits"] (.tuple []),
     .forIn (.name "digits") (.combinationsIterable [poolExpression, .constant (.int 4)]) candidateBody [],
     .return (.name "best_digits")] := rfl

/-- The combination pool includes every nonnegative operand through the signed bound. -/
theorem pool_result (bound : Int) (heap : Mutable.Heap) (env : HeapCalls.Env) (globals : Mutable.Globals)
    (getBound : env "max_digit" = some (Heap.number bound)) :
    ∃ extra, Evaluates program env poolExpression heap (.ref heap.length)
      (working heap ((operands bound).map Heap.number) extra) globals := by
  have stop : Evaluates program env endpoint heap (Heap.number (bound + 1)) heap globals :=
    Evaluates.binary (Evaluates.lookup getBound) Evaluates.constant (by simp [Mutable.binary, scalarBinary])
  by_cases nonnegative : 0 ≤ bound + 1
  · obtain ⟨extra, collected⟩ := range_map (globals := globals) program "$formalsw.range.item"
      (.name "$formalsw.range.item") (fun n => Heap.number (n : Int))
      (fun heap env n found => ⟨[], by simpa using
        (Evaluates.lookup (module := program) (globals := globals) (heap := heap) found)⟩)
      heap env [endpoint] [Heap.number (bound + 1)] 0 (bound + 1).toNat (.cons stop .nil)
      (by change Except.ok (ε := ExceptionObject) ((0 : Int), bound + 1, (1 : Int)) =
            .ok (0, ((0 + (bound + 1).toNat : Nat) : Int), 1)
          simp [Int.toNat_of_nonneg nonnegative])
    exact ⟨extra, by simpa only [operands, List.map_map, Function.comp_def, List.range_eq_range'] using collected⟩
  · have empty : (bound + 1).toNat = 0 := by omega
    refine ⟨[], ?_⟩
    simp only [operands, empty, List.range_zero, List.map_nil]
    apply Evaluates.list_comp_range (values := [Heap.number (bound + 1)]) (.cons stop .nil) rfl
    exact BlockContinues.range_done (by simp [Calls.rangeContinues]; omega) .nil

/-- The complete optimized source returns the same first maximum as the exhaustive reference. -/
theorem result (bound : Int) (heap : Mutable.Heap) (globals : Mutable.Globals)
    (choices : ChoiceSchedule) (anchors : Nat) (stack : List HeapCalls.Frame) (used : Nat) :
    ∃ address extra finalUsed,
      SetCalls.Reaches choices program anchors
        ⟨⟨.operation (.call "solve") [Heap.number bound], stack, heap⟩, used⟩
        ⟨⟨.value (.ref address), stack, heap ++ extra⟩, finalUsed⟩ (globals := globals) ∧
      (heap ++ extra)[address]? = some (.tuple ((answer bound).map Heap.number)) := by
  let args := ((findFunctionIn program.body "solve").getD ([], [])).1
  let source := ((findFunctionIn program.body "solve").getD ([], [])).2
  let entry := functionEnv args source [Heap.number bound] globals
  let scored := setLocal entry "best_length" (Heap.number (-1))
  obtain ⟨emptyAddress, emptyExtra, allocated, foundEmpty⟩ := allocate_tuple [] heap
  have emptyTuple : Evaluates program scored (.tuple []) heap (.ref emptyAddress) (heap ++ emptyExtra) globals := by
    intro anchors stack
    exact .next rfl (.next (by simp [argumentsState, step, HeapCalls.result, run, primitive, allocated]) .refl)
  let initialized := setLocal scored "best_digits" (.ref emptyAddress)
  have initialReady : Ready none (heap ++ emptyExtra) initialized :=
    ⟨by simp [initialized, scored, setLocal, bestScore], emptyAddress,
      by simp [initialized, setLocal], foundEmpty⟩
  have getBound : initialized "max_digit" = some (Heap.number bound) := by cases globals <;> rfl
  obtain ⟨poolExtra, pooled⟩ := pool_result bound (heap ++ emptyExtra) initialized globals getBound
  let afterPool := working (heap ++ emptyExtra) ((operands bound).map Heap.number) poolExtra
  have captured : combinationRows [.ref (heap ++ emptyExtra).length, Heap.number 4] afterPool =
      some (.ok (permutationSources (candidates bound)), afterPool) := by
    have captured := combinationRows_list afterPool (heap ++ emptyExtra).length 4
      ((operands bound).map Heap.number) (found (heap ++ emptyExtra) _ poolExtra)
    simpa only [candidates, permutationSources, Combinations.rows_map, List.map_map, Function.comp_def] using captured
  obtain ⟨searchExtra, finalEnv, finalUsed, searched, finalReady⟩ := candidate_loop (candidates bound) none afterPool
    initialized globals (initialReady.append (.list ((operands bound).map Heap.number) :: poolExtra))
    choices [.return (.name "best_digits")] anchors stack used
  obtain ⟨address, getBest, foundBest⟩ := finalReady.digits
  refine ⟨address, emptyExtra ++ (.list ((operands bound).map Heap.number) :: poolExtra) ++ searchExtra,
    finalUsed, ?_, ?_⟩
  · apply SetCalls.Reaches.next (following := ⟨⟨.block (executableBody source) entry, stack, heap⟩, used⟩)
    · rfl
    · have prepared := (BlockContinues.assign (name := "best_length") (module := program) (env := entry) (globals := globals)
        (Evaluates.unary Evaluates.constant (show HeapCalls.unary .uSub (Heap.number 1) heap =
          some (.ok (Heap.number (-1)), heap) from rfl))).append (BlockContinues.assign (name := "best_digits") emptyTuple)
      apply (SetCalls.Reaches.of_heap (prepared
        [.forIn (.name "digits") (.combinationsIterable [poolExpression, .constant (.int 4)]) candidateBody [],
          .return (.name "best_digits")] anchors stack) choices used).trans
      apply SetCalls.Reaches.next rfl
      apply (SetCalls.Reaches.of_heap (Evaluates.arguments_reach
        (.cons pooled (.cons Evaluates.constant .nil)) _ [] anchors stack) choices used).trans
      apply SetCalls.Reaches.next (following := ⟨⟨.block
        [.permutationStepLoop "digits" (permutationSources (candidates bound)) candidateBody [],
          .return (.name "best_digits")] initialized, stack, afterPool⟩, used⟩)
      · simp only [SetCalls.step, HeapCalls.step, HeapCalls.run, List.nil_append,
          SetCalls.liftStep, comprehensionTargetName, comprehensionTargetBody]
        rw [captured]
        rfl
      · have finished := searched.trans (SetCalls.Reaches.of_heap
          ((BlockReturns.return_value (Evaluates.lookup getBest)) anchors stack) choices finalUsed)
        simpa only [afterPool, working, List.append_assoc] using finished
  · simpa only [afterPool, working, List.append_assoc,
      data.«project-euler-statements».data.py.project_euler_statements.p93.naive.scan_argmax, answer] using foundBest

/-- The omitted argument enters the same proved search at the declared default. -/
theorem default_result (heap : Mutable.Heap) (globals : Mutable.Globals)
    (choices : ChoiceSchedule) (anchors : Nat) (stack : List HeapCalls.Frame) (used : Nat) :
    ∃ address extra finalUsed,
      SetCalls.Reaches choices program anchors
        ⟨⟨.operation (.call "solve") [], stack, heap⟩, used⟩
        ⟨⟨.value (.ref address), stack, heap ++ extra⟩, finalUsed⟩ (globals := globals) ∧
      (heap ++ extra)[address]? = some (.tuple ((answer 9).map Heap.number)) := by
  obtain ⟨address, extra, finalUsed, executed, found⟩ := result 9 heap globals choices anchors stack used
  refine ⟨address, extra, finalUsed, ?_, found⟩
  have same : SetCalls.step (globals := globals) program anchors choices
      ⟨⟨.operation (.call "solve") [], stack, heap⟩, used⟩ =
    SetCalls.step (globals := globals) program anchors choices
      ⟨⟨.operation (.call "solve") [Heap.number 9], stack, heap⟩, used⟩ := rfl
  cases executed with
  | next transition rest => exact .next (same.trans transition) rest

end solvers.«93».solve
