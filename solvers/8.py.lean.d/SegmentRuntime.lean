import solvers.«8.py.lean.d».ScanRuntime

namespace solvers.«8».segmentRuntime
open FormalSW.PyAst HeapCalls.Runtime scanRuntime
set_option maxRecDepth 4096
set_option maxHeartbeats 1200000
private abbrev program := Solvers.P8Ast.module

/-- The initial window product performs twenty runtime units per element. -/
theorem product_bound (digits : List Int) (heap : Mutable.Heap) (address start stop : Nat)
    (env : HeapCalls.Env) (product : Int)
    (found : heap[address]? = some (.list (ints digits)))
    (receiver : env "digits" = some (.ref address))
    (initial : env "prod" = some (.scalar (.int product)))
    (ordered : start ≤ stop) (bounded : stop ≤ digits.length) :
    ∃ finalEnv product',
      BlockContinuesIn .word program env [.rangeStepLoop "t" start stop 1 productBody []]
        heap (20*(stop-start)+4) finalEnv heap ∧
      finalEnv "prod" = some (.scalar (.int product')) ∧
      ∀ name, name ≠ "prod" → name ≠ "t" → finalEnv name = env name := by
  by_cases atEnd : start = stop
  · subst start
    refine ⟨env, product, ?_, initial, by intros; rfl⟩
    simp only [Nat.sub_self, Nat.mul_zero, Nat.zero_add]
    exact BlockContinuesIn.range_done (by simp [Calls.rangeContinues]) BlockContinuesIn.nil
  · have before : start < stop := by omega
    have valid : start < digits.length := by omega
    let nextEnv := HeapCalls.setLocal (HeapCalls.setLocal env "t" (.scalar (.int start)))
      "prod" (.scalar (.int (product * digits[start])))
    obtain ⟨finalEnv, product', tail, result, kept⟩ :=
      product_bound digits heap address (start+1) stop nextEnv (product * digits[start]) found
        (by simp [nextEnv, HeapCalls.setLocal, receiver]) (by simp [nextEnv, HeapCalls.setLocal])
        (by omega) bounded
    refine ⟨finalEnv, product', ?_, result, ?_⟩
    · have count : 20*(stop-start)+4 = (16+(20*(stop-(start+1))+4))+4 := by omega
      rw [count]
      apply BlockContinuesIn.range_step (by simp [Calls.rangeContinues, before])
      apply BlockContinuesIn.append (middleEnv := nextEnv)
      · apply BlockContinuesIn.augAssign (cost := 10) (work := 1) (left := .int product) (right := .int digits[start])
        · simpa [HeapCalls.setLocal] using initial
        · exact read_digit digits heap address start _ "t" found valid
            (by simp [HeapCalls.setLocal, receiver]) (by simp [HeapCalls.setLocal])
        · simp [HeapCalls.inplace, Mutable.binary, scalarBinary]
        · rfl
      · simpa [Calls.loopContinuation, Calls.loopControl, productBody] using tail
    · intro name notProduct notIndex
      rw [kept name notProduct notIndex]
      simp [nextEnv, HeapCalls.setLocal, notProduct, notIndex]
termination_by stop-start

private theorem update_bound (env : HeapCalls.Env) (heap : Mutable.Heap) (product best : Int)
    (current : env "prod" = some (.scalar (.int product)))
    (previous : env "best" = some (.scalar (.int best))) :
    ∃ cost finalEnv best', cost ≤ 12 ∧
      BlockContinuesIn .word program env [updateBest] heap cost finalEnv heap ∧
      finalEnv "best" = some (.scalar (.int best')) ∧
      ∀ name, name ≠ "best" → finalEnv name = env name := by
  have test : EvaluatesIn .word program env (.compare (.name "prod") [(.gt, .name "best")]) heap 7
      (.scalar (.bool (decide (best < product)))) heap :=
    EvaluatesIn.compare (EvaluatesIn.lookup current) (EvaluatesIn.lookup previous)
      (by simp [HeapCalls.compare, scalarCompare, evalCmpOp]) rfl
  by_cases improve : best < product
  · refine ⟨12, HeapCalls.setLocal env "best" (.scalar (.int product)), product, by omega, ?_, rfl, ?_⟩
    · apply BlockContinuesIn.branch (cc := 7) (cb := 3) (value := .bool true) (flag := true)
      · simpa [improve] using test
      · rfl
      · exact BlockContinuesIn.assign (EvaluatesIn.lookup current)
    · intro name different
      simp [HeapCalls.setLocal, different]
  · refine ⟨9, env, best, by omega, ?_, previous, by intros; rfl⟩
    apply BlockContinuesIn.branch (cc := 7) (cb := 0) (value := .bool false) (flag := false)
    · simpa [improve] using test
    · rfl
    · exact BlockContinuesIn.nil

private theorem rolling_value (digits : List Int) (heap : Mutable.Heap) (address index width : Nat)
    (env : HeapCalls.Env) (product : Int)
    (found : heap[address]? = some (.list (ints digits)))
    (receiver : env "digits" = some (.ref address))
    (position : env "t" = some (.scalar (.int index)))
    (size : env "k" = some (.scalar (.int width)))
    (current : env "prod" = some (.scalar (.int product)))
    (within : index < digits.length) (enough : width ≤ index)
    (nonzero : digits[index-width]'(by omega) ≠ 0) :
    EvaluatesIn .word program env rollExpression heap 34
      (.scalar (.int (product.fdiv digits[index-width] * digits[index]))) heap := by
  apply EvaluatesIn.binary (ca := 20) (cb := 10) (work := 1)
      (av := .int (product.fdiv digits[index-width])) (bv := .int digits[index])
  · apply EvaluatesIn.binary (ca := 1) (cb := 15) (work := 1) (bv := .int digits[index-width]) (EvaluatesIn.lookup current)
    · apply EvaluatesIn.list_subscript (ca := 1) (cb := 6) (offset := (index-width : Nat))
        (by intros; simp) (EvaluatesIn.lookup receiver)
      · exact EvaluatesIn.binary (op := .sub) (value := .scalar (.int (index-width : Nat)))
          (EvaluatesIn.lookup position) (EvaluatesIn.lookup size)
          (by simp [Mutable.binary, scalarBinary]; omega) rfl
      · exact found
      · simpa [ints] using Mutable.getItem_list heap address (index-width) (ints digits) found (by simp [ints]; omega)
    · exact divide _ _ heap nonzero
    · rfl
  · exact read_digit digits heap address index env "t" found within receiver position
  · rfl
  · rfl

/-- Each rolling iteration, including updating the maximum, costs at most 52. -/
theorem rolling_bound (digits : List Int) (heap : Mutable.Heap) (address width start stop : Nat)
    (env : HeapCalls.Env) (product best : Int)
    (found : heap[address]? = some (.list (ints digits)))
    (receiver : env "digits" = some (.ref address))
    (size : env "k" = some (.scalar (.int width)))
    (current : env "prod" = some (.scalar (.int product)))
    (previous : env "best" = some (.scalar (.int best)))
    (enough : width ≤ start) (ordered : start ≤ stop) (bounded : stop ≤ digits.length)
    (nonzero : ∀ index, start ≤ index → (bound : index < stop) → digits[index-width]'(by omega) ≠ 0) :
    ∃ cost finalEnv product' best', cost ≤ 52*(stop-start)+4 ∧
      BlockContinuesIn .word program env [.rangeStepLoop "t" start stop 1 rollingBody []] heap cost finalEnv heap ∧
      finalEnv "prod" = some (.scalar (.int product')) ∧
      finalEnv "best" = some (.scalar (.int best')) ∧
      ∀ name, name ≠ "prod" → name ≠ "t" → name ≠ "best" → finalEnv name = env name := by
  by_cases atEnd : start = stop
  · subst start
    exact ⟨4, env, product, best, by omega,
      BlockContinuesIn.range_done (by simp [Calls.rangeContinues]) BlockContinuesIn.nil,
      current, previous, by intros; rfl⟩
  · have before : start < stop := by omega
    have valid : start < digits.length := by omega
    let posEnv := HeapCalls.setLocal env "t" (.scalar (.int start))
    let rolled := product.fdiv digits[start-width] * digits[start]
    let productEnv := HeapCalls.setLocal posEnv "prod" (.scalar (.int rolled))
    have expression : EvaluatesIn .word program posEnv rollExpression heap 34 (.scalar (.int rolled)) heap :=
      rolling_value digits heap address start width posEnv product found
        (by simp [posEnv, HeapCalls.setLocal, receiver]) (by simp [posEnv, HeapCalls.setLocal])
        (by simp [posEnv, HeapCalls.setLocal, size]) (by simp [posEnv, HeapCalls.setLocal, current])
        valid enough (nonzero start (Nat.le_refl _) before)
    obtain ⟨updateCost, bestEnv, nextBest, updateCostBound, update, bestValue, updateKept⟩ :=
      update_bound productEnv heap rolled best rfl
        (by simp [productEnv, posEnv, HeapCalls.setLocal, previous])
    have kept : ∀ name, name ≠ "prod" → name ≠ "t" → name ≠ "best" → bestEnv name = env name := by
      intro name np nt nb
      rw [updateKept name nb]
      simp [productEnv, posEnv, HeapCalls.setLocal, np, nt]
    have productValue : bestEnv "prod" = some (.scalar (.int rolled)) := by
      rw [updateKept "prod" (by decide)]; rfl
    obtain ⟨tailCost, finalEnv, product', best', tailBound, tail, finalProduct, finalBest, tailKept⟩ :=
      rolling_bound digits heap address width (start+1) stop bestEnv rolled nextBest found
        (by rw [kept "digits" (by decide) (by decide) (by decide)]; exact receiver)
        (by rw [kept "k" (by decide) (by decide) (by decide)]; exact size)
        productValue bestValue (by omega) (by omega) bounded
        (fun index lower upper => nonzero index (by omega) upper)
    refine ⟨(36+updateCost)+tailCost+4, finalEnv, product', best', by omega, ?_, finalProduct, finalBest, ?_⟩
    · apply BlockContinuesIn.range_step (by simp [Calls.rangeContinues, before])
      apply BlockContinuesIn.append (middleEnv := bestEnv)
      · exact (BlockContinuesIn.assign expression).append update
      · simpa [Calls.loopContinuation, Calls.loopControl, Calls.loopControlStmt, rollingBody, updateBest] using tail
    · intro name np nt nb
      rw [tailKept name np nt nb, kept name np nt nb]
termination_by stop-start

/-- Initializing and sliding one window costs linearly in its zero-free segment. -/
theorem segment_bound (digits : List Int) (heap : Mutable.Heap) (address start stop width : Nat)
    (env : HeapCalls.Env) (best : Int)
    (found : heap[address]? = some (.list (ints digits)))
    (receiver : env "digits" = some (.ref address))
    (size : env "k" = some (.scalar (.int width)))
    (left : env "i" = some (.scalar (.int start)))
    (right : env "j" = some (.scalar (.int stop)))
    (previous : env "best" = some (.scalar (.int best)))
    (enough : start+width ≤ stop) (bounded : stop ≤ digits.length)
    (nonzero : ∀ index, start ≤ index → (bound : index < stop) → digits[index]'(by omega) ≠ 0) :
    ∃ cost finalEnv best', cost ≤ 52*(stop-start)+51 ∧
      BlockContinuesIn .word program env segmentBody heap cost finalEnv heap ∧
      finalEnv "best" = some (.scalar (.int best')) ∧
      ∀ name, name ≠ "prod" → name ≠ "t" → name ≠ "best" → finalEnv name = env name := by
  let initialEnv := HeapCalls.setLocal env "prod" (.scalar (.int 1))
  obtain ⟨productEnv, product, multiplied, productValue, productKept⟩ :=
    product_bound digits heap address start (start+width) initialEnv 1 found
      (by simp [initialEnv, HeapCalls.setLocal, receiver]) rfl (by omega) (by omega)
  obtain ⟨updateCost, bestEnv, nextBest, updateBound, updated, bestValue, bestKept⟩ :=
    update_bound productEnv heap product best productValue
      (by rw [productKept "best" (by decide) (by decide)]; simpa [initialEnv, HeapCalls.setLocal] using previous)
  have kept : ∀ name, name ≠ "prod" → name ≠ "t" → name ≠ "best" → bestEnv name = env name := by
    intro name np nt nb
    rw [bestKept name nb, productKept name np nt]
    simp [initialEnv, HeapCalls.setLocal, np]
  obtain ⟨rollCost, finalEnv, _, best', rollBound, rolled, _, result, rollKept⟩ :=
    rolling_bound digits heap address width (start+width) stop bestEnv product nextBest found
      (by rw [kept "digits" (by decide) (by decide) (by decide)]; exact receiver)
      (by rw [kept "k" (by decide) (by decide) (by decide)]; exact size)
      (by rw [bestKept "prod" (by decide)]; exact productValue) bestValue
      (by omega) enough bounded
      (fun index lower upper => nonzero (index-width) (by omega) (by omega))
  refine ⟨3+((1+6+(20*((start+width)-start)+4)+7)+(updateCost+(6+1+rollCost+7))),
    finalEnv, best', by omega, ?_, result, ?_⟩
  · apply BlockContinuesIn.append (body := [.assign [.name "prod"] (.constant (.int 1))])
      (BlockContinuesIn.assign EvaluatesIn.constant)
    apply BlockContinuesIn.append (body := [.forIn (.name "t")
      (.call (.name "range") [.name "i", .binOp (.name "i") .add (.name "k")]) productBody []])
    · apply BlockContinuesIn.for_range_pair (ca := 1) (cb := 6) (start := start) (stop := start+width)
        (a := .name "i") (b := .binOp (.name "i") .add (.name "k"))
      · exact EvaluatesIn.lookup (by simpa [initialEnv, HeapCalls.setLocal] using left)
      · exact EvaluatesIn.binary (op := .add) (env := initialEnv) (av := .int start) (bv := .int width)
          (value := .scalar (.int (start+width : Nat))) (middle := heap) (afterRight := heap)
          (EvaluatesIn.lookup (by simpa [initialEnv, HeapCalls.setLocal] using left))
          (EvaluatesIn.lookup (by simpa [initialEnv, HeapCalls.setLocal] using size))
          (by simp [Mutable.binary, scalarBinary]) rfl
      · exact multiplied
    · apply BlockContinuesIn.append updated
      apply BlockContinuesIn.for_range_pair (ca := 6) (cb := 1) (start := start+width) (stop := stop)
        (env := bestEnv) (a := .binOp (.name "i") .add (.name "k")) (b := .name "j")
      · exact EvaluatesIn.binary (op := .add) (env := bestEnv) (av := .int start) (bv := .int width)
          (value := .scalar (.int (start+width : Nat))) (middle := heap) (afterRight := heap)
          (EvaluatesIn.lookup (by rw [kept "i" (by decide) (by decide) (by decide)]; exact left))
          (EvaluatesIn.lookup (by rw [kept "k" (by decide) (by decide) (by decide)]; exact size))
          (by simp [Mutable.binary, scalarBinary]) rfl
      · exact EvaluatesIn.lookup (by rw [kept "j" (by decide) (by decide) (by decide)]; exact right)
      · exact rolled
  · intro name np nt nb
    rw [rollKept name np nt nb, kept name np nt nb]

end solvers.«8».segmentRuntime
