import solvers.«93.py.lean.d».ConsecutiveResult
import data.«project-euler-statements».data.py.project_euler_statements.«p93.py.lean.d».Search

namespace solvers.«93».solve
open FormalSW.PyAst HeapCalls ProjectEulerSolutions.RationalPair
open ProjectEulerSolutions.HeapCollection
open data.«project-euler-statements».data.py.project_euler_statements.p93.consecutive_for_digits (permutationSources)
open data.«project-euler-statements».data.py.project_euler_statements.p93.naive (bestScore nextBest scan)
set_option Elab.async false

abbrev program := Solvers.P93Ast.module
abbrev runLength := all_results.Reference.runLength

abbrev recordBody : List Stmt :=
  [.assign [.name "best_length"] (.name "length"), .assign [.name "best_digits"] (.name "digits")]
abbrev selectBody : List Stmt :=
  [.ifThen (.compare (.name "length") [(.gt, .name "best_length")]) recordBody []]
abbrev candidateBody : List Stmt :=
  .assign [.tuple [.name "length", .name "_"]] (.call (.name "consecutive_length") [.name "digits"]) :: selectBody

structure Ready (best : Option (List Int)) (heap : Mutable.Heap) (env : HeapCalls.Env) : Prop where
  score : env "best_length" = some (Heap.number (bestScore best))
  digits : ∃ address, env "best_digits" = some (.ref address) ∧
    heap[address]? = some (.tuple ((best.getD []).map Heap.number))

theorem Ready.append {best : Option (List Int)} {heap : Mutable.Heap} {env : HeapCalls.Env}
    (ready : Ready best heap env) (extra : Mutable.Heap) : Ready best (heap ++ extra) env := by
  obtain ⟨address, digits, found⟩ := ready.digits
  exact ⟨ready.score, address, digits, prefix_found heap extra address _ found⟩

/-- A strict score improvement replaces the winner; equal scores retain the earlier tuple. -/
theorem select_result (best : Option (List Int)) (candidate : List Int) (address : Nat)
    (heap : Mutable.Heap) (env : HeapCalls.Env) (globals : Mutable.Globals)
    (ready : Ready best heap env)
    (getLength : env "length" = some (Heap.number (runLength candidate)))
    (getDigits : env "digits" = some (.ref address))
    (input : heap[address]? = some (.tuple (candidate.map Heap.number))) :
    ∃ finalEnv, BlockContinues program env selectBody heap finalEnv heap globals ∧
      Ready (nextBest best candidate) heap finalEnv := by
  have test : Evaluates program env (.compare (.name "length") [(.gt, .name "best_length")])
      heap (.scalar (.bool (decide (bestScore best < (runLength candidate : Int))))) heap globals :=
    Evaluates.compare (Evaluates.lookup getLength) (Evaluates.lookup ready.score)
      (by simp [HeapCalls.compare, scalarCompare, evalCmpOp])
  by_cases improved : bestScore best < (runLength candidate : Int)
  · let afterScore := setLocal env "best_length" (Heap.number (runLength candidate))
    let afterDigits := setLocal afterScore "best_digits" (.ref address)
    refine ⟨afterDigits, ?_, ?_⟩
    · apply BlockContinues.branch (flag := true) test (by simp [improved, Mutable.truth, scalarTruth])
      exact (BlockContinues.assign (Evaluates.lookup getLength)).append
        (BlockContinues.assign (Evaluates.lookup (by simpa [afterScore, setLocal] using getDigits)))
    · rw [nextBest, if_pos improved]
      exact ⟨by simp [afterDigits, afterScore, setLocal, bestScore], address,
        by simp [afterDigits, setLocal], input⟩
  · refine ⟨env, BlockContinues.branch (flag := false) test
      (by simp [improved, Mutable.truth, scalarTruth]) .nil, ?_⟩
    simpa only [nextBest, if_neg improved] using ready

/-- Score one candidate, unpack the actual result tuple, and retain the first strict maximum. -/
theorem candidate_result (best : Option (List Int)) (candidate : List Int) (address : Nat)
    (heap : Mutable.Heap) (env : HeapCalls.Env) (globals : Mutable.Globals)
    (ready : Ready best heap env) (getDigits : env "digits" = some (.ref address))
    (input : heap[address]? = some (.tuple (candidate.map Heap.number)))
    (choices : ChoiceSchedule) (following : List Stmt) (anchors : Nat) (stack : List HeapCalls.Frame) (used : Nat) :
    ∃ extra finalEnv finalUsed,
      SetCalls.Reaches choices program anchors
        ⟨⟨.block (candidateBody ++ following) env, stack, heap⟩, used⟩
        ⟨⟨.block following finalEnv, stack, heap ++ extra⟩, finalUsed⟩ (globals := globals) ∧
      Ready (nextBest best candidate) (heap ++ extra) finalEnv := by
  obtain ⟨extra, tupleAddress, setAddress, finalUsed, foundTuple, counted⟩ :=
    consecutive_length.result candidate heap address globals input choices anchors
      (.unpack ["length", "_"] (selectBody ++ following) env :: stack) used
  let afterScore := assignNames env ["length", "_"] [Heap.number (runLength candidate), .ref setAddress]
  obtain ⟨bestAddress, getBest, foundBest⟩ := ready.digits
  have prepared : Ready best (heap ++ extra) afterScore :=
    ⟨by simpa [afterScore, assignNames, setLocal] using ready.score, bestAddress,
      by simpa [afterScore, assignNames, setLocal] using getBest,
      prefix_found heap extra bestAddress _ foundBest⟩
  obtain ⟨finalEnv, selected, finalReady⟩ := select_result best candidate address (heap ++ extra) afterScore globals
    prepared (by simp [afterScore, assignNames, setLocal])
    (by simpa [afterScore, assignNames, setLocal] using getDigits) (prefix_found heap extra address _ input)
  refine ⟨extra, finalEnv, finalUsed, ?_, finalReady⟩
  apply SetCalls.Reaches.next (following := ⟨⟨.expr (.call (.name "consecutive_length") [.name "digits"]) env,
    .unpack ["length", "_"] (selectBody ++ following) env :: stack, heap⟩, used⟩) rfl
  apply SetCalls.Reaches.next rfl
  apply (SetCalls.Reaches.of_heap (Evaluates.arguments_reach
    (.cons (Evaluates.lookup getDigits) .nil) (.call "consecutive_length") [] anchors _) choices used).trans
  apply counted.trans
  apply SetCalls.Reaches.next (following := ⟨⟨.block (selectBody ++ following) afterScore, stack, heap ++ extra⟩, finalUsed⟩)
  · simp [SetCalls.step, HeapCalls.step, HeapCalls.run, Mutable.sequence, foundTuple, SetCalls.liftStep, afterScore]
  · exact SetCalls.Reaches.of_heap (selected following anchors stack) choices finalUsed

/-- Every combination is visited once, with arbitrary private set-iteration choices inside scoring. -/
theorem candidate_loop (candidates : List (List Int)) (best : Option (List Int)) (heap : Mutable.Heap)
    (env : HeapCalls.Env) (globals : Mutable.Globals) (ready : Ready best heap env)
    (choices : ChoiceSchedule) (following : List Stmt) (anchors : Nat) (stack : List HeapCalls.Frame) (used : Nat) :
    ∃ extra finalEnv finalUsed,
      SetCalls.Reaches choices program anchors
        ⟨⟨.block (.permutationStepLoop "digits" (permutationSources candidates) candidateBody [] :: following)
          env, stack, heap⟩, used⟩
        ⟨⟨.block following finalEnv, stack, heap ++ extra⟩, finalUsed⟩ (globals := globals) ∧
      Ready (scan candidates best) (heap ++ extra) finalEnv := by
  induction candidates generalizing best heap env used with
  | nil => exact ⟨[], env, used, by simpa using SetCalls.Reaches.next (module := program) rfl .refl, by simpa [scan] using ready⟩
  | cons candidate candidates ih =>
    obtain ⟨bestAddress, getBest, foundBest⟩ := ready.digits
    obtain ⟨address, tupleExtra, allocated, foundTuple⟩ := allocate_tuple (candidate.map Heap.number) heap
    let bound := setLocal env "digits" (.ref address)
    have boundReady : Ready best (heap ++ tupleExtra) bound :=
      ⟨by simpa [bound, setLocal] using ready.score, bestAddress,
        by simpa [bound, setLocal] using getBest, prefix_found heap tupleExtra bestAddress _ foundBest⟩
    obtain ⟨scoreExtra, after, afterUsed, scored, afterReady⟩ := candidate_result best candidate address
      (heap ++ tupleExtra) bound globals boundReady (by simp [bound, setLocal]) foundTuple choices
      (.permutationStepLoop "digits" (permutationSources candidates) candidateBody [] :: following) anchors stack used
    obtain ⟨later, finalEnv, finalUsed, continued, finalReady⟩ :=
      ih (nextBest best candidate) ((heap ++ tupleExtra) ++ scoreExtra) after afterReady afterUsed
    refine ⟨tupleExtra ++ scoreExtra ++ later, finalEnv, finalUsed, ?_, ?_⟩
    · apply SetCalls.Reaches.next (following := ⟨⟨.block
        (candidateBody ++ .permutationStepLoop "digits" (permutationSources candidates) candidateBody [] :: following)
        bound, stack, heap ++ tupleExtra⟩, used⟩)
      · simp [SetCalls.step, HeapCalls.step, HeapCalls.run, permutationSources, List.map_map, Function.comp_def,
          sequenceValue, sequenceSource, allocated, SetCalls.liftStep, bound,
          Calls.loopContinuation, Calls.loopControl, Calls.loopControlStmt, candidateBody, selectBody, recordBody]
      · simpa only [List.append_assoc] using scored.trans continued
    · simpa only [scan, List.foldl_cons, List.append_assoc] using finalReady

end solvers.«93».solve
