import solvers.«68.pruning.source»

namespace solvers.«68»
open FormalSW.PyAst HeapCalls ProjectEulerStatements MagicRing MagicRingTextCalls
set_option Elab.async false

def stableLocals : List String := ["n", "digits", "total_digits", "total", "labels", "nums"]

structure OuterReady (n : Nat) (digits : Int) (best : String) (labelsAddress numsAddress : Nat)
    (env : HeapCalls.Env) : Prop where
  getN : env "n" = some (number n)
  getDigits : env "digits" = some (.scalar (.int digits))
  getBest : env "best" = some (string best)
  getWidth : env "total_digits" = some (number ((labels n).map digitWidth).sum)
  getTotal : env "total" = some (number (n * (2 * n + 1)))
  getLabels : env "labels" = some (.ref labelsAddress)
  getNums : env "nums" = some (.ref numsAddress)

theorem OuterReady.of_kept
    (ready : OuterReady n digits best labelsAddress numsAddress before)
    (getBest : after "best" = some (string nextBest))
    (kept : ∀ name ∈ stableLocals, after name = before name) :
    OuterReady n digits nextBest labelsAddress numsAddress after where
  getN := (kept "n" (by decide)).trans ready.getN
  getDigits := (kept "digits" (by decide)).trans ready.getDigits
  getBest := getBest
  getWidth := (kept "total_digits" (by decide)).trans ready.getWidth
  getTotal := (kept "total" (by decide)).trans ready.getTotal
  getLabels := (kept "labels" (by decide)).trans ready.getLabels
  getNums := (kept "nums" (by decide)).trans ready.getNums

theorem OuterReady.set_auxiliary
    (ready : OuterReady n digits best labelsAddress numsAddress env)
    (name : String) (value : Mutable.Value) (auxiliary : name ∉ "best" :: stableLocals) :
    OuterReady n digits best labelsAddress numsAddress (setLocal env name value) := by
  apply ready.of_kept
  · have different : "best" ≠ name := by
      intro same
      subst name
      exact auxiliary List.mem_cons_self
    simpa only [setLocal, different, if_false] using ready.getBest
  · intro key member
    have different : key ≠ name := by
      intro same
      subst key
      exact auxiliary (List.mem_cons_of_mem _ member)
    simp only [setLocal, different, if_false]

theorem OuterReady.permutation_kept
    (ready : OuterReady n digits best labelsAddress numsAddress before)
    (getBest : after "best" = some (string nextBest)) (kept : PermutationKeeps before after) :
    OuterReady n digits nextBest labelsAddress numsAddress after := by
  apply ready.of_kept getBest
  intro name member
  simp only [stableLocals, List.mem_cons, List.not_mem_nil, or_false] at member
  rcases member with rfl | rfl | rfl | rfl | rfl | rfl <;>
    exact kept _ (by decide) (by decide) (by decide) (by decide)
      (by decide) (by decide) (by decide)

theorem combination_result (n : Nat) (positive : 0 < n) (digits : Int) (best : String)
    (selected : List Nat) (size : selected.length = n) (distinct : selected.Nodup)
    (subset : ∀ value ∈ selected, value ∈ labels n) (heap : Mutable.Heap) (env : HeapCalls.Env)
    (labelsAddress numsAddress selectedAddress : Nat)
    (getNums : heap[numsAddress]? = some (.set (IntegerSets.entries (ints (labels n)))))
    (getSelected : heap[selectedAddress]? = some (.tuple (encode selected)))
    (localSelected : env "inner_set" = some (.ref selectedAddress))
    (ready : OuterReady n digits best labelsAddress numsAddress env)
    (next : Stmt) (following : List Stmt) (answer : Mutable.Value)
    (post : ∀ nextEnv extra,
      OuterReady n digits (selectionVisit n digits best selected)
        labelsAddress numsAddress nextEnv →
      ∃ finalHeap, BlockReturns program nextEnv (next :: following) (heap ++ extra)
        answer finalHeap) :
    ∃ finalHeap, BlockReturns program env
      (combinationBody ++ [Calls.loopContinuation combinationBody next] ++ following)
      heap answer finalHeap := by
  have widthTest := width_condition_result n selected digits heap env selectedAddress
    getSelected localSelected ready.getWidth ready.getDigits
  rw [show Calls.loopContinuation combinationBody next = .loopResume next from rfl,
    combination_code_source]
  by_cases width : (requiredDigits n selected : Int) = digits
  · let summed := setLocal env "inner_sum" (number selected.sum)
    have sumReady : OuterReady n digits best labelsAddress numsAddress summed :=
      ready.set_auxiliary "inner_sum" _ (by decide)
    have divTest := divisibility_condition_result n positive selected heap summed
      sumReady.getTotal rfl sumReady.getN
    by_cases divisible : (n * (2 * n + 1) + selected.sum) % n = 0
    · have admitted : SelectionAdmissible n digits selected := ⟨width, divisible⟩
      have visit : selectionVisit n digits best selected =
          innerScan n digits (Permutations.rows n selected) best := by
        rw [selectionVisit, if_pos admitted]
      let targeted := setLocal summed "target" (number (target n selected))
      have targetReady : OuterReady n digits best labelsAddress numsAddress targeted :=
        sumReady.set_auxiliary "target" _ (by decide)
      let allocations : Mutable.Heap :=
        [.set (IntegerSets.entries (ints selected)),
         .set (ScalarSet.difference (IntegerSets.entries (ints (labels n)))
           (IntegerSets.entries (ints selected)))]
      let preparedHeap := heap ++ allocations
      let prepared := setLocal targeted "remaining_outer" (.ref (heap.length + 1))
      have preparedReady : OuterReady n digits best labelsAddress numsAddress prepared :=
        targetReady.set_auxiliary "remaining_outer" _ (by decide)
      have complement : Evaluates program targeted complementExpression heap
          (.ref (heap.length + 1)) preparedHeap := by
        exact complement_result heap targeted (ints (labels n)) (ints selected)
          numsAddress selectedAddress getNums
          (by simpa only [IntegerSets.encoded, ints, encode, List.map_map, Function.comp_def]
            using getSelected) targetReady.getNums
          (by simpa [targeted, summed, setLocal] using localSelected)
      obtain ⟨finalHeap, scanned⟩ := inner_loop_result n positive digits selected size distinct
        subset best preparedHeap prepared selectedAddress (heap.length + 1)
        (ProjectEulerSolutions.HeapCollection.prefix_found heap allocations _ _ getSelected)
        (by simp [preparedHeap, allocations])
        (by simpa [prepared, targeted, summed, setLocal] using localSelected) rfl
        preparedReady.getN (by simp [prepared, targeted, setLocal])
        preparedReady.getDigits preparedReady.getBest (.loopResume next :: following) answer (by
          intro finalEnv extra finalBest kept
          have finalReady : OuterReady n digits (selectionVisit n digits best selected)
              labelsAddress numsAddress finalEnv := preparedReady.permutation_kept
            (by simpa only [visit] using finalBest) kept
          obtain ⟨finalHeap, continued⟩ := post finalEnv (allocations ++ extra) finalReady
          exact ⟨finalHeap, BlockReturns.loop_resume
            (by simpa only [preparedHeap, List.append_assoc] using continued)⟩)
      refine ⟨finalHeap, ?_⟩
      apply BlockReturns.branch (flag := false)
        (by simpa only [ne_eq, width, not_true_eq_false, decide_false] using widthTest) rfl
      apply BlockReturns.assign (inner_sum_result selected heap env selectedAddress
        getSelected localSelected)
      apply BlockReturns.branch (flag := false)
        (by simpa only [ne_eq, divisible, not_true_eq_false, decide_false] using divTest) rfl
      apply BlockReturns.assign (target_expression_result n positive selected heap summed
        sumReady.getTotal rfl sumReady.getN)
      apply BlockReturns.assign complement
      exact scanned
    · have visit : selectionVisit n digits best selected = best := by
        simp only [selectionVisit, SelectionAdmissible, divisible, and_false, if_false]
      obtain ⟨finalHeap, continued⟩ := post summed [] (by simpa only [visit] using sumReady)
      refine ⟨finalHeap, ?_⟩
      apply BlockReturns.branch (flag := false)
        (by simpa only [ne_eq, width, not_true_eq_false, decide_false] using widthTest) rfl
      apply BlockReturns.assign (inner_sum_result selected heap env selectedAddress
        getSelected localSelected)
      apply BlockReturns.branch (flag := true)
        (by simpa only [ne_eq, divisible, not_false_eq_true, decide_true] using divTest) rfl
      exact BlockReturns.continue_loop (next := next) (following := following) rfl
        (by simpa only [List.append_nil] using continued)
  · have visit : selectionVisit n digits best selected = best := by
      simp only [selectionVisit, SelectionAdmissible, width, false_and, if_false]
    obtain ⟨finalHeap, continued⟩ := post env [] (by simpa only [visit] using ready)
    refine ⟨finalHeap, ?_⟩
    apply BlockReturns.branch (flag := true)
      (by simpa only [ne_eq, width, not_false_eq_true, decide_true] using widthTest) rfl
    exact BlockReturns.continue_loop (next := next) (following := following) rfl
      (by simpa only [List.append_nil] using continued)

end solvers.«68»
