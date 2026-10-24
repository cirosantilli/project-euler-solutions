import solvers.«68.sets.source»
import ProjectEulerStatements.Lib.MagicRingOptimizedScan

namespace solvers.«68»
open FormalSW.PyAst HeapCalls ProjectEulerStatements MagicRing MagicRingTextCalls
set_option Elab.async false

def CandidateKeeps (before after : HeapCalls.Env) : Prop :=
  ∀ name, name ≠ "outer" → name ≠ "index" → name ≠ "value" →
    name ≠ "start" → name ≠ "text" → name ≠ "best" → after name = before name

abbrev optimizedTextExpression : Expr :=
  .call (.name "ring_string") [.name "outer", .name "inner", .name "start"]
abbrev incompleteExpression : Expr :=
  .compare (.builtinCall .len [.name "outer"]) [(.notEq, .name "n")]

theorem accepted_code_source : innerPermutationBody.drop 4 =
    [.assign [.name "start"] minimumExpression,
     .assign [.name "text"] optimizedTextExpression] ++ MagicRingRecordCalls.code := rfl

theorem candidate_code_source : innerPermutationBody = outerCode ++
    [.ifThen incompleteExpression [.continueLoop] [],
     .ifThen validationExpression [.continueLoop] [],
     .assign [.name "start"] minimumExpression,
     .assign [.name "text"] optimizedTextExpression] ++ MagicRingRecordCalls.code := rfl

theorem candidate_tail_source : innerPermutationBody.drop 2 =
    [.ifThen incompleteExpression [.continueLoop] [],
     .ifThen validationExpression [.continueLoop] []] ++ innerPermutationBody.drop 4 := rfl

theorem accepted_result {n : Nat} (ring : Ring n) (positive : 0 < n)
    (digits : Int) (best : String) (heap : Mutable.Heap) (env : HeapCalls.Env)
    (outerAddress innerAddress : Nat)
    (getOuter : heap[outerAddress]? = some (.list (encode ring.outer)))
    (getInner : heap[innerAddress]? = some (.tuple (encode ring.inner)))
    (localOuter : env "outer" = some (.ref outerAddress))
    (localInner : env "inner" = some (.ref innerAddress))
    (getN : env "n" = some (number n))
    (getDigits : env "digits" = some (.scalar (.int digits)))
    (getBest : env "best" = some (string best)) :
    ∃ finalEnv extra, BlockContinues program env (innerPermutationBody.drop 4) heap
      finalEnv (heap ++ extra) ∧
      finalEnv "best" = some (string (optimizedVisit ring digits best)) ∧
      CandidateKeeps env finalEnv := by
  obtain ⟨pooled, minimized⟩ := minimum_result ring positive heap env outerAddress
    getOuter localOuter getN
  let selected := setLocal env "start" (number (minimumIndex ring.outer))
  let represented := textFrom ring.outer ring.inner (minimumIndex ring.outer)
  let textEnv := setLocal selected "text" (string represented)
  let parts := pieces ring (minimumIndex ring.outer) n
  have outerFound : (heap ++ pooled)[outerAddress]? = some (.list (encode ring.outer)) := by
    rw [List.getElem?_append_left (List.getElem?_eq_some_iff.mp getOuter).1]
    exact getOuter
  have innerFound : (heap ++ pooled)[innerAddress]? = some (.tuple (encode ring.inner)) := by
    rw [List.getElem?_append_left (List.getElem?_eq_some_iff.mp getInner).1]
    exact getInner
  have serialized : Evaluates program selected optimizedTextExpression (heap ++ pooled)
      (string represented) (working (heap ++ pooled) parts) := by
    exact Evaluates.call
      (.cons (Evaluates.lookup (by simpa [selected, setLocal] using localOuter))
        (.cons (Evaluates.lookup (by simpa [selected, setLocal] using localInner))
          (.cons (Evaluates.lookup rfl) .nil)))
      (ring_string.result ring (minimumIndex ring.outer) (heap ++ pooled)
        outerAddress innerAddress outerFound innerFound)
  obtain ⟨finalEnv, recorded, finalBest, kept⟩ := MagicRingRecordCalls.result program
    represented best digits (working (heap ++ pooled) parts) textEnv rfl
    (by simpa [textEnv, selected, setLocal] using getBest)
    (by simpa [textEnv, selected, setLocal] using getDigits)
  refine ⟨finalEnv, pooled ++ [.list (encodedStrings parts)], ?_, finalBest, ?_⟩
  · rw [accepted_code_source]
    simpa only [working, List.append_assoc] using
      (BlockContinues.assign minimized).append ((BlockContinues.assign serialized).append recorded)
  · intro name notOuter notIndex notValue notStart notText notBest
    rw [kept name notBest]
    simp [textEnv, selected, setLocal, notStart, notText]

theorem incomplete_result (n : Nat) (outer : List Int) (heap : Mutable.Heap)
    (env : HeapCalls.Env) (outerAddress : Nat)
    (getOuter : heap[outerAddress]? = some (.list (IntegerSets.encoded outer)))
    (localOuter : env "outer" = some (.ref outerAddress))
    (getN : env "n" = some (number n)) :
    Evaluates program env incompleteExpression heap
      (.scalar (.bool (decide (outer.length ≠ n)))) heap := by
  apply Evaluates.compare (middle := heap) (afterRight := heap)
    (av := number outer.length) (bv := number n)
  · exact Evaluates.builtin (.cons (Evaluates.lookup localOuter) .nil)
      (by simp [HeapCalls.primitive, Mutable.length, getOuter, IntegerSets.encoded])
  · exact Evaluates.lookup getN
  · simp [HeapCalls.compare, scalarCompare, evalCmpOp]

theorem after_outer_result (n : Nat) (positive : 0 < n) (digits : Int) (best : String)
    (inner : List Nat) (size : inner.length = n) (distinct : inner.Nodup)
    (subset : ∀ value ∈ inner, value ∈ labels n) (heap : Mutable.Heap) (env : HeapCalls.Env)
    (outerAddress innerAddress remainingAddress : Nat)
    (getOuter : heap[outerAddress]? =
      some (.list (IntegerSets.encoded (builtOuter n inner))))
    (getInner : heap[innerAddress]? = some (.tuple (encode inner)))
    (getRemaining : heap[remainingAddress]? = some (.set
      (ScalarSet.difference (IntegerSets.entries (ints (labels n)))
        (IntegerSets.entries (ints inner)))))
    (localOuter : env "outer" = some (.ref outerAddress))
    (localInner : env "inner" = some (.ref innerAddress))
    (localRemaining : env "remaining_outer" = some (.ref remainingAddress))
    (getN : env "n" = some (number n))
    (getDigits : env "digits" = some (.scalar (.int digits)))
    (getBest : env "best" = some (string best))
    (next : Stmt) (following : List Stmt) (answer : Mutable.Value)
    (post : ∀ nextEnv extra,
      nextEnv "best" = some (string (innerVisit n digits best inner)) →
      CandidateKeeps env nextEnv →
      ∃ finalHeap, BlockReturns program nextEnv (next :: following) (heap ++ extra)
        answer finalHeap) :
    ∃ finalHeap, BlockReturns program env
      (innerPermutationBody.drop 2 ++ [.loopResume next] ++ following) heap answer finalHeap := by
  have incomplete := incomplete_result n (builtOuter n inner) heap env outerAddress
    getOuter localOuter getN
  rw [candidate_tail_source]
  by_cases full : (builtOuter n inner).length = n
  · obtain ⟨garbage, tested⟩ := validation_result n (builtOuter n inner) _ heap env
      outerAddress remainingAddress getOuter getRemaining localOuter localRemaining getN
    cases flag : validationFlag n (builtOuter n inner)
        (ScalarSet.difference (IntegerSets.entries (ints (labels n)))
          (IntegerSets.entries (ints inner))) with
    | false =>
        obtain ⟨same, cardinality⟩ := (validation_flag_false _ _ _).mp flag
        obtain ⟨ring, inside, outerEq, selected⟩ := validated_ring n inner size distinct subset
          full cardinality same
        have ringOuter : heap[outerAddress]? = some (.list (encode ring.outer)) := by
          simpa only [outerEq, ints, IntegerSets.encoded, encode, List.map_map,
            Function.comp_def] using getOuter
        have extendedOuter : (heap ++ garbage)[outerAddress]? =
            some (.list (encode ring.outer)) := by
          rw [List.getElem?_append_left (List.getElem?_eq_some_iff.mp ringOuter).1]
          exact ringOuter
        have extendedInner : (heap ++ garbage)[innerAddress]? =
            some (.tuple (encode ring.inner)) := by
          rw [List.getElem?_append_left (List.getElem?_eq_some_iff.mp getInner).1]
          simpa only [inside] using getInner
        obtain ⟨finalEnv, extra, accepted, finalBest, kept⟩ := accepted_result ring positive
          digits best (heap ++ garbage) env outerAddress innerAddress extendedOuter
          extendedInner localOuter localInner getN getDigits getBest
        have sameVisit : innerVisit n digits best inner = optimizedVisit ring digits best :=
          innerVisit_ring n digits best inner ring inside ((mem_optimizedRings ring).mp selected)
        obtain ⟨finalHeap, continued⟩ := post finalEnv (garbage ++ extra)
          (by simpa only [sameVisit] using finalBest) kept
        refine ⟨finalHeap, ?_⟩
        apply BlockReturns.branch (flag := false)
          (by simpa only [full, ne_eq, not_true_eq_false, decide_false] using incomplete) rfl
        apply BlockReturns.branch (flag := false) (by simpa only [flag] using tested) rfl
        exact accepted.returns (BlockReturns.loop_resume
          (by simpa only [List.append_assoc] using continued))
    | true =>
        have rejected : ¬ (ScalarSet.compare .eq (IntegerSets.entries (builtOuter n inner))
            (ScalarSet.difference (IntegerSets.entries (ints (labels n)))
              (IntegerSets.entries (ints inner))) = true ∧
            (IntegerSets.entries (builtOuter n inner)).length = n) := by
          intro valid
          have impossible := (validation_flag_false _ _ _).mpr valid
          rw [flag] at impossible
          contradiction
        have unchanged := innerVisit_rejected n positive digits best inner rejected
        obtain ⟨finalHeap, continued⟩ := post env garbage
          (by simpa only [unchanged] using getBest) (fun _ _ _ _ _ _ _ => rfl)
        refine ⟨finalHeap, ?_⟩
        apply BlockReturns.branch (flag := false)
          (by simpa only [full, ne_eq, not_true_eq_false, decide_false] using incomplete) rfl
        apply BlockReturns.branch (flag := true) (by simpa only [flag] using tested) rfl
        exact BlockReturns.continue_loop (next := next) (following := following) rfl continued
  · have unchanged := innerVisit_incomplete n positive digits best inner full
    obtain ⟨finalHeap, continued⟩ := post env []
      (by simpa only [unchanged] using getBest) (fun _ _ _ _ _ _ _ => rfl)
    refine ⟨finalHeap, ?_⟩
    apply BlockReturns.branch (flag := true)
      (by simpa only [ne_eq, full, not_false_eq_true, decide_true] using incomplete) rfl
    exact BlockReturns.continue_loop (next := next) (following := following) rfl
      (by simpa only [List.append_nil] using continued)

theorem candidate_result (n : Nat) (positive : 0 < n) (digits : Int) (best : String)
    (inner : List Nat) (size : inner.length = n) (distinct : inner.Nodup)
    (subset : ∀ value ∈ inner, value ∈ labels n) (heap : Mutable.Heap) (env : HeapCalls.Env)
    (innerAddress remainingAddress : Nat)
    (getInner : heap[innerAddress]? = some (.tuple (encode inner)))
    (getRemaining : heap[remainingAddress]? = some (.set
      (ScalarSet.difference (IntegerSets.entries (ints (labels n)))
        (IntegerSets.entries (ints inner)))))
    (localInner : env "inner" = some (.ref innerAddress))
    (localRemaining : env "remaining_outer" = some (.ref remainingAddress))
    (getN : env "n" = some (number n))
    (getTarget : env "target" = some (number (target n inner)))
    (getDigits : env "digits" = some (.scalar (.int digits)))
    (getBest : env "best" = some (string best))
    (next : Stmt) (following : List Stmt) (answer : Mutable.Value)
    (post : ∀ nextEnv extra,
      nextEnv "best" = some (string (innerVisit n digits best inner)) →
      CandidateKeeps env nextEnv →
      ∃ finalHeap, BlockReturns program nextEnv (next :: following) (heap ++ extra)
        answer finalHeap) :
    ∃ finalHeap, BlockReturns program env
      (innerPermutationBody ++ [Calls.loopContinuation innerPermutationBody next] ++ following)
      heap answer finalHeap := by
  obtain ⟨constructedEnv, constructed, localOuter, kept⟩ := outer_result n inner size
    (target n inner) heap env innerAddress getInner localInner getN getTarget
  let built := builtOuter n inner
  let constructedHeap := MagicRingOuterCalls.working heap built
  have outerFound : constructedHeap[heap.length]? =
      some (.list (IntegerSets.encoded built)) := by
    simp [constructedHeap, MagicRingOuterCalls.working, MagicRingOuterCalls.encodeInts,
      IntegerSets.encoded]
  have innerFound : constructedHeap[innerAddress]? = some (.tuple (encode inner)) :=
    MagicRingOuterCalls.prefix_found heap built innerAddress _ getInner
  have remainingFound : constructedHeap[remainingAddress]? = some (.set
      (ScalarSet.difference (IntegerSets.entries (ints (labels n)))
        (IntegerSets.entries (ints inner)))) :=
    MagicRingOuterCalls.prefix_found heap built remainingAddress _ getRemaining
  obtain ⟨finalHeap, continued⟩ := after_outer_result n positive digits best inner size
    distinct subset constructedHeap constructedEnv heap.length innerAddress remainingAddress
    outerFound innerFound remainingFound localOuter
    (by rw [kept "inner" (by decide) (by decide) (by decide)]; exact localInner)
    (by rw [kept "remaining_outer" (by decide) (by decide) (by decide)]; exact localRemaining)
    (by rw [kept "n" (by decide) (by decide) (by decide)]; exact getN)
    (by rw [kept "digits" (by decide) (by decide) (by decide)]; exact getDigits)
    (by rw [kept "best" (by decide) (by decide) (by decide)]; exact getBest)
    next following answer (by
      intro nextEnv extra finalBest laterKept
      have allKept : CandidateKeeps env nextEnv := by
        intro name notOuter notIndex notValue notStart notText notBest
        rw [laterKept name notOuter notIndex notValue notStart notText notBest,
          kept name notOuter notIndex notValue]
      obtain ⟨finalHeap, finished⟩ := post nextEnv
        (.list (MagicRingOuterCalls.encodeInts built) :: extra) finalBest allKept
      exact ⟨finalHeap, by simpa only [constructedHeap, MagicRingOuterCalls.working,
        List.append_assoc, List.singleton_append] using finished⟩)
  refine ⟨finalHeap, ?_⟩
  change BlockReturns program env
    (innerPermutationBody ++ [.loopResume next] ++ following) heap answer finalHeap
  have divided : innerPermutationBody =
      innerPermutationBody.take 2 ++ innerPermutationBody.drop 2 :=
    (List.take_append_drop 2 innerPermutationBody).symm
  rw [divided, List.append_assoc]
  exact constructed.returns continued

end solvers.«68»
