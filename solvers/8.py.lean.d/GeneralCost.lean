import solvers.«8.py.lean.d».Source
import lang.py.FormalSW.HeapReadCost

/-! Weighted costs for arbitrary digit lists and window widths. The read-only
source phases preserve the heap, so their allocation-weighted cost counts every
transition and does not conceal sequence copies. -/
namespace solvers.«8».generalCost
open FormalSW.PyAst
open HeapCalls.AllocationCost.ReadOnly
set_option maxRecDepth 4096
set_option maxHeartbeats 1200000
private abbrev program := Solvers.P8Ast.module

private theorem read_digit (digits : List Int) (heap : Mutable.Heap) (address position : Nat)
    (env : HeapCalls.Env) (indexName : String)
    (found : heap[address]? = some (.list (ints digits))) (valid : position < digits.length)
    (receiver : env "digits" = some (.ref address))
    (index : env indexName = some (.scalar (.int position))) :
    EvaluatesIn program env (.subscript (.name "digits") (.name indexName)) heap 6
      (.scalar (.int digits[position])) heap := by
  apply EvaluatesIn.subscript (by intros; simp) (EvaluatesIn.lookup receiver) (EvaluatesIn.lookup index)
  simpa [ints] using Mutable.getItem_list heap address position (ints digits) found (by simpa [ints] using valid)

private theorem scan_end (digits : List Int) (heap : Mutable.Heap) (position : Nat)
    (env : HeapCalls.Env) (name : String) (zeros : Bool)
    (index : env name = some (.scalar (.int position)))
    (count : env "digit_count" = some (.scalar (.int digits.length))) (finished : digits.length ≤ position) :
    EvaluatesIn program env (scanTest name zeros) heap 7 (.scalar (.bool false)) heap := by
  apply EvaluatesIn.boolean_stop (value := .scalar (.bool false)) (flag := false)
  · exact EvaluatesIn.compare (EvaluatesIn.lookup index) (EvaluatesIn.lookup count)
      (by simp [HeapCalls.compare, scalarCompare, evalCmpOp]; omega)
  · rfl
  · rfl

private theorem scan_read (digits : List Int) (heap : Mutable.Heap) (address position : Nat)
    (env : HeapCalls.Env) (name : String) (zeros : Bool)
    (found : heap[address]? = some (.list (ints digits))) (valid : position < digits.length)
    (receiver : env "digits" = some (.ref address))
    (index : env name = some (.scalar (.int position)))
    (count : env "digit_count" = some (.scalar (.int digits.length))) :
    EvaluatesIn program env (scanTest name zeros) heap 18
      (.scalar (.bool (decide (digits[position] = 0) == zeros))) heap := by
  apply EvaluatesIn.boolean_pair (ca := 5) (cb := 10) (value := .scalar (.bool true))
      (flag := true) (answerFlag := decide (digits[position] = 0) == zeros)
  · exact EvaluatesIn.compare (EvaluatesIn.lookup index) (EvaluatesIn.lookup count)
      (by simp [HeapCalls.compare, scalarCompare, evalCmpOp, valid])
  · rfl
  · rfl
  · apply EvaluatesIn.compare (read_digit digits heap address position env name found valid receiver index)
      EvaluatesIn.constant
    cases zeros <;> simp [HeapCalls.compare, scalarCompare, evalCmpOp]
  · rfl

/-- Both scans cost at most 26 transitions per consumed digit plus 20 to stop. -/
theorem scan_bound (digits : List Int) (heap : Mutable.Heap) (address start stop : Nat)
    (env : HeapCalls.Env) (name : String) (zeros : Bool)
    (found : heap[address]? = some (.list (ints digits)))
    (receiver : env "digits" = some (.ref address))
    (count : env "digit_count" = some (.scalar (.int digits.length)))
    (notDigits : "digits" ≠ name) (notCount : "digit_count" ≠ name)
    (ordered : start ≤ stop) (bounded : stop ≤ digits.length)
    (selected : ∀ index, start ≤ index → (bound : index < stop) →
      (decide (digits[index]'(by omega) = 0) == zeros) = true)
    (finished : ∀ bound : stop < digits.length,
      (decide (digits[stop]'bound = 0) == zeros) = false) :
    ∃ cost, cost ≤ 26*(stop-start)+20 ∧
      BlockContinuesIn program (HeapCalls.setLocal env name (.scalar (.int start)))
        [scanLoop name zeros] heap cost (HeapCalls.setLocal env name (.scalar (.int stop))) heap := by
  by_cases atEnd : start = stop
  · subst start
    by_cases valid : stop < digits.length
    · refine ⟨20, by omega, ?_⟩
      apply BlockContinuesIn.while_loop (cc := 18) (cb := 0) (value := .scalar (.bool false)) (flag := false)
      · simpa only [finished valid] using scan_read digits heap address stop
          (HeapCalls.setLocal env name (.scalar (.int stop))) name zeros found valid
          (by simp [HeapCalls.setLocal, notDigits, receiver])
          (by simp [HeapCalls.setLocal]) (by simp [HeapCalls.setLocal, notCount, count])
      · rfl
      · exact BlockContinuesIn.nil
    · refine ⟨9, by omega, ?_⟩
      apply BlockContinuesIn.while_loop (cc := 7) (cb := 0) (value := .scalar (.bool false)) (flag := false)
      · exact scan_end digits heap stop _ name zeros
          (by simp [HeapCalls.setLocal]) (by simp [HeapCalls.setLocal, notCount, count]) (by omega)
      · rfl
      · exact BlockContinuesIn.nil
  · have before : start < stop := by omega
    obtain ⟨cost, bound, tail⟩ := scan_bound digits heap address (start+1) stop env name zeros found receiver count
      notDigits notCount (by omega) bounded (fun index lower upper => selected index (by omega) upper) finished
    refine ⟨18+(6+cost)+2, by omega, ?_⟩
    apply BlockContinuesIn.while_loop (cc := 18) (cb := 6+cost) (value := .scalar (.bool true)) (flag := true)
    · simpa only [selected start (by omega) before] using scan_read digits heap address start
        (HeapCalls.setLocal env name (.scalar (.int start))) name zeros found (by omega)
        (by simp [HeapCalls.setLocal, notDigits, receiver])
        (by simp [HeapCalls.setLocal]) (by simp [HeapCalls.setLocal, notCount, count])
    · rfl
    · apply BlockContinuesIn.append (middleEnv := HeapCalls.setLocal env name (.scalar (.int (start+1 : Nat))))
      · have increment := BlockContinuesIn.augAssign (module := program)
          (env := HeapCalls.setLocal env name (.scalar (.int start))) (name := name) (heap := heap)
          (op := .add) (left := .scalar (.int start)) (right := .scalar (.int 1))
          (value := .scalar (.int (start+1 : Nat))) (by simp [HeapCalls.setLocal]) EvaluatesIn.constant
          (by simp [HeapCalls.inplace, Mutable.binary, scalarBinary])
        simpa only [setLocal_shadow] using increment
      · exact tail
termination_by stop-start

/-- The initial window product performs twelve transitions per element. -/
theorem product_bound (digits : List Int) (heap : Mutable.Heap) (address start stop : Nat)
    (env : HeapCalls.Env) (product : Int)
    (found : heap[address]? = some (.list (ints digits)))
    (receiver : env "digits" = some (.ref address))
    (initial : env "prod" = some (.scalar (.int product)))
    (ordered : start ≤ stop) (bounded : stop ≤ digits.length) :
    ∃ finalEnv product',
      BlockContinuesIn program env [.rangeStepLoop "t" start stop 1 productBody []]
        heap (12*(stop-start)+1) finalEnv heap ∧
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
    · have count : 12*(stop-start)+1 = (11+(12*(stop-(start+1))+1))+1 := by omega
      rw [count]
      apply BlockContinuesIn.range_step (by simp [Calls.rangeContinues, before])
      apply BlockContinuesIn.append (middleEnv := nextEnv)
      · apply BlockContinuesIn.augAssign (cost := 6) (left := .scalar (.int product)) (right := .scalar (.int digits[start]))
        · simpa [HeapCalls.setLocal] using initial
        · exact read_digit digits heap address start _ "t" found valid
            (by simp [HeapCalls.setLocal, receiver]) (by simp [HeapCalls.setLocal])
        · simp [HeapCalls.inplace, Mutable.binary, scalarBinary]
      · simpa [Calls.loopContinuation, Calls.loopControl, productBody] using tail
    · intro name notProduct notIndex
      rw [kept name notProduct notIndex]
      simp [nextEnv, HeapCalls.setLocal, notProduct, notIndex]
termination_by stop-start

private theorem update_bound (env : HeapCalls.Env) (heap : Mutable.Heap) (product best : Int)
    (current : env "prod" = some (.scalar (.int product)))
    (previous : env "best" = some (.scalar (.int best))) :
    ∃ cost finalEnv best', cost ≤ 10 ∧
      BlockContinuesIn program env [updateBest] heap cost finalEnv heap ∧
      finalEnv "best" = some (.scalar (.int best')) ∧
      ∀ name, name ≠ "best" → finalEnv name = env name := by
  have test : EvaluatesIn program env (.compare (.name "prod") [(.gt, .name "best")]) heap 5
      (.scalar (.bool (decide (best < product)))) heap :=
    EvaluatesIn.compare (EvaluatesIn.lookup current) (EvaluatesIn.lookup previous)
      (by simp [HeapCalls.compare, scalarCompare, evalCmpOp])
  by_cases improve : best < product
  · refine ⟨10, HeapCalls.setLocal env "best" (.scalar (.int product)), product, by omega, ?_, rfl, ?_⟩
    · apply BlockContinuesIn.branch (cc := 5) (cb := 3) (value := .scalar (.bool true)) (flag := true)
      · simpa [improve] using test
      · rfl
      · exact BlockContinuesIn.assign (EvaluatesIn.lookup current)
    · intro name different
      simp [HeapCalls.setLocal, different]
  · refine ⟨7, env, best, by omega, ?_, previous, by intros; rfl⟩
    apply BlockContinuesIn.branch (cc := 5) (cb := 0) (value := .scalar (.bool false)) (flag := false)
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
    EvaluatesIn program env rollExpression heap 23
      (.scalar (.int (product.fdiv digits[index-width] * digits[index]))) heap := by
  apply EvaluatesIn.binary (ca := 14) (cb := 6)
      (av := .scalar (.int (product.fdiv digits[index-width]))) (bv := .scalar (.int digits[index]))
  · apply EvaluatesIn.binary (ca := 1) (cb := 10) (bv := .scalar (.int digits[index-width])) (EvaluatesIn.lookup current)
    · apply EvaluatesIn.subscript (cr := 1) (ci := 5) (position := .scalar (.int (index-width : Nat)))
        (by intros; simp) (EvaluatesIn.lookup receiver)
      · exact EvaluatesIn.binary (EvaluatesIn.lookup position) (EvaluatesIn.lookup size)
          (by simp [Mutable.binary, scalarBinary]; omega)
      · simpa [ints] using Mutable.getItem_list heap address (index-width) (ints digits) found (by simp [ints]; omega)
    · exact divide _ _ heap nonzero
  · exact read_digit digits heap address index env "t" found within receiver position
  · rfl

/-- Each rolling iteration, including updating the maximum, costs at most 36. -/
theorem rolling_bound (digits : List Int) (heap : Mutable.Heap) (address width start stop : Nat)
    (env : HeapCalls.Env) (product best : Int)
    (found : heap[address]? = some (.list (ints digits)))
    (receiver : env "digits" = some (.ref address))
    (size : env "k" = some (.scalar (.int width)))
    (current : env "prod" = some (.scalar (.int product)))
    (previous : env "best" = some (.scalar (.int best)))
    (enough : width ≤ start) (ordered : start ≤ stop) (bounded : stop ≤ digits.length)
    (nonzero : ∀ index, start ≤ index → (bound : index < stop) → digits[index-width]'(by omega) ≠ 0) :
    ∃ cost finalEnv product' best', cost ≤ 36*(stop-start)+1 ∧
      BlockContinuesIn program env [.rangeStepLoop "t" start stop 1 rollingBody []] heap cost finalEnv heap ∧
      finalEnv "prod" = some (.scalar (.int product')) ∧
      finalEnv "best" = some (.scalar (.int best')) ∧
      ∀ name, name ≠ "prod" → name ≠ "t" → name ≠ "best" → finalEnv name = env name := by
  by_cases atEnd : start = stop
  · subst start
    exact ⟨1, env, product, best, by omega,
      BlockContinuesIn.range_done (by simp [Calls.rangeContinues]) BlockContinuesIn.nil,
      current, previous, by intros; rfl⟩
  · have before : start < stop := by omega
    have valid : start < digits.length := by omega
    let posEnv := HeapCalls.setLocal env "t" (.scalar (.int start))
    let rolled := product.fdiv digits[start-width] * digits[start]
    let productEnv := HeapCalls.setLocal posEnv "prod" (.scalar (.int rolled))
    have expression : EvaluatesIn program posEnv rollExpression heap 23 (.scalar (.int rolled)) heap :=
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
    refine ⟨(25+updateCost)+tailCost+1, finalEnv, product', best', by omega, ?_, finalProduct, finalBest, ?_⟩
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
    ∃ cost finalEnv best', cost ≤ 36*(stop-start)+35 ∧
      BlockContinuesIn program env segmentBody heap cost finalEnv heap ∧
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
  refine ⟨3+((8+((12*((start+width)-start)+1)+1)+1)+(updateCost+(8+(rollCost+1)+1))),
    finalEnv, best', by omega, ?_, result, ?_⟩
  · apply BlockContinuesIn.append (body := [.assign [.name "prod"] (.constant (.int 1))])
      (BlockContinuesIn.assign EvaluatesIn.constant)
    apply BlockContinuesIn.append (body := [.forIn (.name "t")
      (.call (.name "range") [.name "i", .binOp (.name "i") .add (.name "k")]) productBody []])
    · apply BlockContinuesIn.for_range (ca := 8) (values := [.scalar (.int start), .scalar (.int (start+width : Nat))])
      · apply EvaluatesIn.Arguments.cons (tailCost := 6)
          (EvaluatesIn.lookup (by simpa [initialEnv, HeapCalls.setLocal] using left))
        apply EvaluatesIn.Arguments.cons (headCost := 5) (tailCost := 0) _ .nil
        exact EvaluatesIn.binary
          (EvaluatesIn.lookup (by simpa [initialEnv, HeapCalls.setLocal] using left))
          (EvaluatesIn.lookup (by simpa [initialEnv, HeapCalls.setLocal] using size))
          (by simp [Mutable.binary, scalarBinary])
      · rfl
      · exact multiplied
    · apply BlockContinuesIn.append updated
      apply BlockContinuesIn.for_range (ca := 8) (values := [.scalar (.int (start+width : Nat)), .scalar (.int stop)])
      · apply EvaluatesIn.Arguments.cons (headCost := 5) (tailCost := 2) _
          (.cons (EvaluatesIn.lookup (by rw [kept "j" (by decide) (by decide) (by decide)]; exact right)) .nil)
        exact EvaluatesIn.binary
          (EvaluatesIn.lookup (by rw [kept "i" (by decide) (by decide) (by decide)]; exact left))
          (EvaluatesIn.lookup (by rw [kept "k" (by decide) (by decide) (by decide)]; exact size))
          (by simp [Mutable.binary, scalarBinary])
      · rfl
      · exact rolled
  · intro name np nt nb
    rw [rollKept name np nt nb, kept name np nt nb]

end solvers.«8».generalCost
