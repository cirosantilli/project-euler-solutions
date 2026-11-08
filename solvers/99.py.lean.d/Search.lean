import solvers.«99.py.lean.d».Comparison
import ProjectEulerSolutions.Lib.PowerMaximum

namespace solvers.«99».solve
open FormalSW.PyAst HeapCalls ProjectEulerSolutions.PowerIntervals ProjectEulerSolutions.PowerMaximum
set_option Elab.async false

abbrev args := ((findFunctionIn program.body "solve").getD ([], [])).1
abbrev source := ((findFunctionIn program.body "solve").getD ([], [])).2
abbrev iteration : List Stmt := match source with
  | [_, _, _, .forIn _ _ body _, _] => body
  | _ => []
abbrev chooseTest : Expr :=
  .boolOp .or [.compare (.name "best_idx") [(.eq, .constant (.int 0))],
    .compare (.call (.name "compare_powers")
      [.name "base", .name "exponent", .name "best_base", .name "best_exp"])
      [(.gt, .constant (.int 0))]]

structure Locals (env : HeapCalls.Env) (address best : Nat) (incumbent : Pair) : Prop where
  getPairs : env "pairs" = some (.ref address)
  getBest : env "best_idx" = some (number best)
  getBase : env "best_base" = some (number incumbent.1)
  getExponent : env "best_exp" = some (number incumbent.2)

theorem choose_result (current incumbent : Pair) (best : Nat)
    (positiveCurrent : 0 < current.1) (positiveIncumbent : 0 < incumbent.1)
    (env : HeapCalls.Env) (heap : Mutable.Heap) (globals : Mutable.Globals)
    (getCurrentBase : env "base" = some (number current.1))
    (getCurrentExponent : env "exponent" = some (number current.2))
    (getBest : env "best_idx" = some (number best))
    (getBase : env "best_base" = some (number incumbent.1))
    (getExponent : env "best_exp" = some (number incumbent.2)) :
    ∃ extra, Evaluates program env chooseTest heap
      (.scalar (.bool (decide (best = 0 ∨ 0 < signedComparison current.1 current.2 incumbent.1 incumbent.2))))
      (heap ++ extra) globals := by
  by_cases empty : best = 0
  · refine ⟨[], ?_⟩
    simp only [empty, true_or, decide_true, List.append_nil]
    apply Evaluates.boolean_stop (flag := true) _ rfl rfl
    apply Evaluates.compare (Evaluates.lookup getBest) Evaluates.constant
    simp [HeapCalls.compare, scalarCompare, evalCmpOp, empty]
  · obtain ⟨extra, compared⟩ := compare_powers.result current.1 current.2 incumbent.1 incumbent.2
      positiveCurrent positiveIncumbent heap globals
    have first : Evaluates program env (.compare (.name "best_idx") [(.eq, .constant (.int 0))])
        heap (.scalar (.bool false)) heap globals :=
      Evaluates.compare (Evaluates.lookup getBest) Evaluates.constant
        (by simp [HeapCalls.compare, scalarCompare, evalCmpOp, empty])
    refine ⟨extra, ?_⟩
    simp only [empty, false_or]
    apply Evaluates.boolean_next (flag := false) first rfl rfl
    apply Evaluates.boolean_last _ rfl
    apply Evaluates.compare (middle := heap ++ extra) (afterRight := heap ++ extra)
    · exact Evaluates.call (.cons (Evaluates.lookup getCurrentBase)
        (.cons (Evaluates.lookup getCurrentExponent)
          (.cons (Evaluates.lookup getBase) (.cons (Evaluates.lookup getExponent) .nil)))) compared
    · exact Evaluates.constant
    · simp [HeapCalls.compare, scalarCompare, evalCmpOp]

theorem iteration_result (current incumbent : Pair) (next best address : Nat)
    (positiveCurrent : 0 < current.1) (positiveIncumbent : 0 < incumbent.1)
    (env : HeapCalls.Env) (heap : Mutable.Heap) (globals : Mutable.Globals)
    (locals : Locals env address best incumbent)
    (getIndex : env "index" = some (number next))
    (getCurrentBase : env "base" = some (number current.1))
    (getCurrentExponent : env "exponent" = some (number current.2)) :
    ∃ readyEnv extra, BlockContinues program env iteration heap readyEnv (heap ++ extra) globals ∧
      Locals readyEnv address
        (if best = 0 ∨ 0 < signedComparison current.1 current.2 incumbent.1 incumbent.2 then next else best)
        (if best = 0 ∨ 0 < signedComparison current.1 current.2 incumbent.1 incumbent.2 then current else incumbent) := by
  have valid : Evaluates program env (.compare (.name "base") [(.lt, .constant (.int 1))])
      heap (.scalar (.bool false)) heap globals :=
    Evaluates.compare (Evaluates.lookup getCurrentBase) Evaluates.constant
      (by simp [HeapCalls.compare, scalarCompare, evalCmpOp]; omega)
  obtain ⟨extra, chosen⟩ := choose_result current incumbent best positiveCurrent positiveIncumbent
    env heap globals getCurrentBase getCurrentExponent locals.getBest locals.getBase locals.getExponent
  by_cases choose : best = 0 ∨ 0 < signedComparison current.1 current.2 incumbent.1 incumbent.2
  · let readyEnv := setLocal (setLocal (setLocal env "best_idx" (number next))
      "best_base" (number current.1)) "best_exp" (number current.2)
    refine ⟨readyEnv, extra, ?_, ?_⟩
    · apply BlockContinues.append (body := iteration.take 1) (rest := iteration.drop 1)
        (middleEnv := env) (middleHeap := heap)
      · exact BlockContinues.branch (flag := false) valid rfl BlockContinues.nil
      · apply BlockContinues.branch (flag := true) (by simpa [choose] using chosen) rfl
        apply BlockContinues.append (body := [.assign [.name "best_idx"] (.name "index")])
          (BlockContinues.assign (Evaluates.lookup getIndex))
        apply BlockContinues.append (body := [.assign [.name "best_base"] (.name "base")])
          (BlockContinues.assign (Evaluates.lookup (by simpa [setLocal] using getCurrentBase)))
        exact BlockContinues.assign (Evaluates.lookup (by simpa [setLocal] using getCurrentExponent))
    · simp only [if_pos choose]
      rcases locals with ⟨hp, hi, hb, he⟩
      constructor <;> simp_all [readyEnv, setLocal]
  · refine ⟨env, extra, ?_, ?_⟩
    · apply BlockContinues.append (body := iteration.take 1) (rest := iteration.drop 1)
        (middleEnv := env) (middleHeap := heap)
      · exact BlockContinues.branch (flag := false) valid rfl BlockContinues.nil
      · apply BlockContinues.branch (flag := false) (by simpa [choose] using chosen) rfl
        exact BlockContinues.nil
    · simpa only [if_neg choose] using locals

abbrev bindings : List Stmt :=
  [.assign [.name "index"] (.name "$formalsw.enumerate.index"),
   .assign [.tuple [.name "$formalsw.enumerate.right.0", .name "$formalsw.enumerate.right.1"]]
     (.name "$formalsw.enumerate.value"),
   .assign [.name "base"] (.name "$formalsw.enumerate.right.0"),
   .assign [.name "exponent"] (.name "$formalsw.enumerate.right.1")]

theorem bindings_lowered :
    UnpackingTargets.lower (.name "index") (.name "$formalsw.enumerate.index") "$formalsw.enumerate.left" ++
    UnpackingTargets.lower (.tuple [.name "base", .name "exponent"])
      (.name "$formalsw.enumerate.value") "$formalsw.enumerate.right" = bindings := by
  simp [UnpackingTargets.lower, UnpackingTargets.children, UnpackingTargets.names,
    UnpackingTargets.temporary, List.range_succ, bindings]
  decide

abbrev loop (address position next : Nat) : Stmt :=
  .enumerateStepLoop "$formalsw.enumerate.index" "$formalsw.enumerate.value"
    (.ref address) position next (bindings ++ iteration) []

abbrev Entry := Nat × Pair
abbrev entryValue (entry : Entry) : Mutable.Value := .ref entry.1
abbrev entryObject (entry : Entry) : Mutable.Object :=
  .tuple [number entry.2.1, number entry.2.2]

theorem bindings_result (entry : Entry) (next best address : Nat) (incumbent : Pair)
    (env : HeapCalls.Env) (heap : Mutable.Heap) (globals : Mutable.Globals)
    (locals : Locals env address best incumbent)
    (found : heap[entry.1]? = some (entryObject entry)) :
    ∃ readyEnv, BlockContinues program
        (setLocal (setLocal env "$formalsw.enumerate.index" (number next))
          "$formalsw.enumerate.value" (entryValue entry)) bindings heap readyEnv heap globals ∧
      Locals readyEnv address best incumbent ∧
      readyEnv "index" = some (number next) ∧
      readyEnv "base" = some (number entry.2.1) ∧
      readyEnv "exponent" = some (number entry.2.2) := by
  let bound := setLocal (setLocal env "$formalsw.enumerate.index" (number next))
    "$formalsw.enumerate.value" (entryValue entry)
  let indexed := setLocal bound "index" (number next)
  let unpacked := assignNames indexed ["$formalsw.enumerate.right.0", "$formalsw.enumerate.right.1"]
    [number entry.2.1, number entry.2.2]
  let readyEnv := setLocal (setLocal unpacked "base" (number entry.2.1)) "exponent" (number entry.2.2)
  refine ⟨readyEnv, ?_, ?_, ?_, ?_, ?_⟩
  · apply BlockContinues.append (body := bindings.take 1) (rest := bindings.drop 1)
      (middleEnv := indexed) (middleHeap := heap)
    · exact BlockContinues.assign (Evaluates.lookup (value := number next) (by simp [setLocal]))
    · apply BlockContinues.append (body := (bindings.drop 1).take 1)
        (rest := bindings.drop 2) (middleEnv := unpacked) (middleHeap := heap)
      · apply BlockContinues.unpack (names := ["$formalsw.enumerate.right.0", "$formalsw.enumerate.right.1"])
          (items := [number entry.2.1, number entry.2.2]) (value := entryValue entry) rfl
          (Evaluates.lookup (name := "$formalsw.enumerate.value") (by simp [indexed, bound, setLocal]))
          _ rfl
        simp [Mutable.sequence, found, entryObject]
      · apply BlockContinues.append (body := (bindings.drop 2).take 1) (rest := bindings.drop 3)
          (middleEnv := setLocal unpacked "base" (number entry.2.1)) (middleHeap := heap)
        · exact BlockContinues.assign (Evaluates.lookup (by simp [unpacked, assignNames, setLocal]))
        · exact BlockContinues.assign (Evaluates.lookup (by simp [unpacked, assignNames, setLocal]))
  · rcases locals with ⟨hp, hi, hb, he⟩
    constructor <;> simp_all [readyEnv, unpacked, indexed, bound, assignNames, setLocal]
  all_goals simp [readyEnv, unpacked, indexed, bound, assignNames, setLocal]

theorem loop_result (remaining before : List Entry) (next best address : Nat) (incumbent : Pair)
    (positiveIncumbent : 0 < incumbent.1)
    (env : HeapCalls.Env) (heap : Mutable.Heap) (globals : Mutable.Globals)
    (locals : Locals env address best incumbent)
    (stored : heap[address]? = some (.list ((before ++ remaining).map entryValue)))
    (valid : ∀ entry ∈ remaining, 0 < entry.2.1 ∧ heap[entry.1]? = some (entryObject entry)) :
    ∃ finalEnv extra finalPair,
      BlockContinues program env [loop address before.length next] heap finalEnv (heap ++ extra) globals ∧
      Locals finalEnv address (scan (remaining.map Prod.snd) next best incumbent) finalPair := by
  induction remaining generalizing before next best incumbent env heap with
  | nil =>
      refine ⟨env, [], incumbent, ?_, locals⟩
      simp only [List.append_nil] at stored ⊢
      apply BlockContinues.enumerate_done (items := before.map entryValue)
      · simp [Mutable.sequence, sequenceValue, stored]
      · simp
      · exact BlockContinues.nil
  | cons current rest ih =>
      obtain ⟨positiveCurrent, found⟩ := valid current (by simp)
      obtain ⟨boundEnv, bound, boundLocals, getIndex, getBase, getExponent⟩ :=
        bindings_result current next best address incumbent env heap globals locals found
      obtain ⟨readyEnv, extra, ran, readyLocals⟩ := iteration_result current.2 incumbent next best address
        positiveCurrent positiveIncumbent boundEnv heap globals boundLocals getIndex getBase getExponent
      let choose := best = 0 ∨ 0 < signedComparison current.2.1 current.2.2 incumbent.1 incumbent.2
      have append_found (location : Nat) (object : Mutable.Object) (present : heap[location]? = some object) :
          (heap ++ extra)[location]? = some object := by
        rw [List.getElem?_append_left (List.getElem?_eq_some_iff.mp present).1]
        exact present
      have newStored : (heap ++ extra)[address]? =
          some (.list (((before ++ [current]) ++ rest).map entryValue)) := by
        simpa only [List.append_assoc, List.singleton_append] using append_found address _ stored
      have newValid : ∀ entry ∈ rest, 0 < entry.2.1 ∧ (heap ++ extra)[entry.1]? = some (entryObject entry) := by
        intro entry member
        obtain ⟨positive, object⟩ := valid entry (by simp [member])
        exact ⟨positive, append_found entry.1 _ object⟩
      obtain ⟨finalEnv, tailExtra, finalPair, tailRan, finalLocals⟩ :=
        ih (before ++ [current]) (next + 1) (if choose then next else best)
          (if choose then current.2 else incumbent) (by split <;> assumption)
          readyEnv (heap ++ extra) readyLocals newStored newValid
      refine ⟨finalEnv, extra ++ tailExtra, finalPair, ?_, ?_⟩
      · apply BlockContinues.enumerate_step (value := entryValue current)
          (items := (before ++ current :: rest).map entryValue)
        · simp [Mutable.sequence, sequenceValue, stored]
        · simp
        · have composed := (bound.append ran).append tailRan
          simpa only [loop, Calls.loopContinuation, Calls.loopControl, Calls.loopControlStmt,
            bindings, iteration, source, program, Bool.false_or, Bool.false_eq_true, if_false,
            List.length_append, List.length_singleton, Int.natCast_add, Int.natCast_one,
            List.append_assoc] using composed
      · change Locals finalEnv address
          (if choose then scan (rest.map Prod.snd) (next + 1) next current.2
           else scan (rest.map Prod.snd) (next + 1) best incumbent) finalPair
        by_cases selected : choose <;> simpa only [selected, ite_true, ite_false] using finalLocals

theorem result (entries : List Entry) (address : Nat) (heap : Mutable.Heap)
    (stored : heap[address]? = some (.list (entries.map entryValue)))
    (valid : ∀ entry ∈ entries, 0 < entry.2.1 ∧ heap[entry.1]? = some (entryObject entry))
    (globals : Mutable.Globals := []) :
    ∃ extra, ReturnsValue program "solve" [.ref address] heap
      (number (maximumIndex (entries.map Prod.snd))) (heap ++ extra) globals := by
  let initial := functionEnv args source [.ref address] globals
  let ready := setLocal (setLocal (setLocal initial "best_idx" (number 0))
    "best_base" (number 1)) "best_exp" (number 0)
  have locals : Locals ready address 0 (1, 0) := by
    constructor <;> cases globals <;> rfl
  obtain ⟨finalEnv, extra, finalPair, ran, finalLocals⟩ :=
    loop_result entries [] 1 0 address (1, 0) (by decide) ready heap globals locals stored valid
  have flattened : BlockContinues program ready
      [.forIn (.tuple [.name "$formalsw.enumerate.index", .name "$formalsw.enumerate.value"])
        (.call (.name "enumerate") [.name "pairs", .constant (.int 1)]) (bindings ++ iteration) []]
      heap finalEnv (heap ++ extra) globals := by
    apply BlockContinues.for_enumerate_start (items := entries.map entryValue)
      (Evaluates.lookup locals.getPairs) Evaluates.constant rfl
    · simp [Mutable.sequence, stored]
    · exact ran
  have enumerated : BlockContinues program ready ((source.drop 3).take 1)
      heap finalEnv (heap ++ extra) globals := by
    intro following anchors stack
    refine .next (following := ⟨.block
      ([.forIn (.tuple [.name "$formalsw.enumerate.index", .name "$formalsw.enumerate.value"])
        (.call (.name "enumerate") [.name "pairs", .constant (.int 1)]) (bindings ++ iteration) []] ++ following)
      ready, stack, heap⟩) ?_ (flattened following anchors stack)
    change step program anchors ⟨.block
      (.forIn (.tuple [.name "index", .tuple [.name "base", .name "exponent"]])
        (.call (.name "enumerate") [.name "pairs", .constant (.int 1)]) iteration [] :: following)
      ready, stack, heap⟩ globals = _
    simp only [step, bindings_lowered, protectGenerator_next, List.singleton_append]
  refine ⟨extra, ?_⟩
  apply BlockReturns.function (args := args) (body := source) rfl rfl
  apply BlockReturns.assign (value := number 0) (middle := heap) Evaluates.constant
  apply BlockReturns.assign (value := number 1) (middle := heap) Evaluates.constant
  apply BlockReturns.assign (value := number 0) (middle := heap) Evaluates.constant
  exact enumerated.returns (BlockReturns.return_value (Evaluates.lookup finalLocals.getBest))

end solvers.«99».solve
