import «.formalsw».solvers.«77.py.ast»
import ProjectEulerStatements.Lib.CoinChange
import ProjectEulerStatements.Lib.PrimePartitions
import lang.py.FormalSW.LoopCalls

/-! The source DP processes precisely the positive-denomination prefix before its first break. -/
namespace solvers.«77»
open FormalSW.PyAst HeapCalls
open ProjectEulerStatements.P31 ProjectEulerStatements.CoinChange
abbrev program := Solvers.P77Ast.module
abbrev number (n : Nat) : Mutable.Value := .scalar (.int n)
abbrev encode (items : List Nat) : List Mutable.Value := items.map number
set_option Elab.async false

namespace count_prime_summations
private def table (size : Nat) (value : Nat → Nat) : List Mutable.Value :=
  (List.range size).map (fun n => number (value n))

private theorem table_lookup (size : Nat) (value : Nat → Nat) (position : Nat) (inside : position < size) :
    (table size value)[position]? = some (number (value position)) := by
  simp [table, inside]

private theorem table_set (size : Nat) (value : Nat → Nat) (position replacement : Nat) (inside : position < size) :
    (table size value).set position (number replacement) =
      table size (fun n => if n = position then replacement else value n) := by
  apply List.ext_getElem?
  intro n
  by_cases bounded : n < size
  · by_cases same : position = n
    · subst n
      rw [List.getElem?_set_self (by simpa [table] using inside), table_lookup size _ position inside]
      simp
    · rw [List.getElem?_set_ne same, table_lookup size value n bounded,
        table_lookup size _ n bounded]
      simp [Ne.symm same]
  · have outside : size ≤ n := by omega
    simp only [List.getElem?_eq_none (by simpa [table] using outside : ((table size value).set position (number replacement)).length ≤ n),
      List.getElem?_eq_none (by simpa [table] using outside : (table size (fun k => if k = position then replacement else value k)).length ≤ n)]

private theorem table_congr (size : Nat) (f g : Nat → Nat) (equal : ∀ n, n < size → f n = g n) :
    table size f = table size g := by
  apply List.map_congr_left
  intro n member
  rw [equal n (List.mem_range.mp member)]

private def workHeap (base : Mutable.Heap) (n : Nat) (value : Nat → Nat) : Mutable.Heap :=
  base ++ [.list (table (n + 1) value)]

private theorem workHeap_store (base : Mutable.Heap) (n : Nat) (f g : Nat → Nat) :
    (workHeap base n f).set base.length (.list (table (n + 1) g)) = workHeap base n g := by
  simp [workHeap, List.set_append_right]

private abbrev innerBody : List Stmt :=
  [.augAssign (.subscript (.name "ways") (.name "s")) .add
    (.subscript (.name "ways") (.binOp (.name "s") .sub (.name "p")))]
private abbrev coinBody : List Stmt :=
  [.forIn (.name "s") (.call (.name "range")
    [.name "p", .binOp (.name "n") .add (.constant (.int 1))]) innerBody []]

private structure Locals (env : HeapCalls.Env) (n primesAddress waysAddress : Nat) : Prop where
  targetValue : env "n" = some (number n)
  primes : env "primes" = some (.ref primesAddress)
  ways : env "ways" = some (.ref waysAddress)

private theorem update_cell (n p s : Nat) (inside : s ≤ n) (lower : p ≤ s)
    (value : Nat → Nat) (base : Mutable.Heap) (env : HeapCalls.Env)
    (getDp : env "ways" = some (.ref base.length))
    (getC : env "p" = some (number p)) (getS : env "s" = some (number s)) :
    BlockContinues program env innerBody (workHeap base n value) env
      (workHeap base n (fun n => if n = s then value s + value (s - p) else value n)) := by
  have found : (workHeap base n value)[base.length]? = some (.list (table (n + 1) value)) := by
    simp [workHeap]
  have readAt (query : Nat) (bound : query ≤ n) :
      Mutable.getItem (.ref base.length) (number query) (workHeap base n value) =
        some (.ok (number (value query)), workHeap base n value) := by
    have entry := table_lookup (n + 1) value query (by omega)
    simpa only [(List.getElem?_eq_some_iff.mp entry).2] using
      Mutable.getItem_list (workHeap base n value) base.length query _ found (by simpa [table] using Nat.lt_succ_of_le bound)
  apply BlockContinues.augItem (by intro a b p; simp) (Evaluates.lookup getDp) (Evaluates.lookup getS)
    (previous := number (value s)) (updated := number (value s + value (s - p)))
    (afterRead := workHeap base n value) (afterOperation := workHeap base n value)
  · exact readAt s inside
  · apply Evaluates.subscript (by intro a b p; simp) (Evaluates.lookup getDp)
      (Evaluates.binary (av := number s) (bv := number p) (value := number (s - p))
        (middle := workHeap base n value) (afterRight := workHeap base n value)
        (finalHeap := workHeap base n value) (Evaluates.lookup getS) (Evaluates.lookup getC)
        (by simp [Mutable.binary, scalarBinary, number, Int.natCast_sub lower]))
    exact readAt (s - p) (by omega)
  · simp [inplace, Mutable.binary, scalarBinary, number]
  · rw [← workHeap_store base n value
      (fun n => if n = s then value s + value (s - p) else value n),
      ← table_set (n + 1) value s (value s + value (s - p)) (by omega)]
    simp [Mutable.setItem, Mutable.index, found, table, number, Except.map,
      show ¬ (s : Int) < 0 by omega, show ¬ n + 1 ≤ s by omega]

/-- During a coin pass, entries below the frontier already allow the new coin. -/
private theorem inner_loop (n p first primesAddress : Nat) (positive : 0 < p) (lower : p ≤ first)
    (processed : List Nat) (base : Mutable.Heap) (env : HeapCalls.Env)
    (locals : Locals env n primesAddress base.length)
    (getC : env "p" = some (number p)) :
    ∃ finalEnv, Locals finalEnv n primesAddress base.length ∧
      BlockContinues program env [.rangeStepLoop "s" first (n + 1) 1 innerBody []]
        (workHeap base n (frontier p processed first)) finalEnv
        (workHeap base n (fun n => countWaysBound n (p :: processed))) := by
  by_cases active : first ≤ n
  · let bound := setLocal env "s" (number first)
    have boundLocals : Locals bound n primesAddress base.length :=
      ⟨by simpa [bound, setLocal] using locals.targetValue,
       by simpa [bound, setLocal] using locals.primes,
       by simpa [bound, setLocal] using locals.ways⟩
    have boundC : bound "p" = some (number p) := by simpa [bound, setLocal] using getC
    obtain ⟨finalEnv, finalLocals, rest⟩ := inner_loop n p (first + 1) primesAddress positive (by omega)
      processed base bound boundLocals boundC
    refine ⟨finalEnv, finalLocals, ?_⟩
    apply BlockContinues.range_step (by simp [Calls.rangeContinues]; omega)
    apply BlockContinues.append (body := innerBody) (middleEnv := bound)
      (middleHeap := workHeap base n (frontier p processed (first + 1)))
    · have updated := update_cell n p first active lower (frontier p processed first) base bound
        boundLocals.ways boundC (by simp [bound, setLocal])
      have same : (fun n => if n = first then frontier p processed first first +
          frontier p processed first (first - p) else frontier p processed first n) =
          frontier p processed (first + 1) := by
        funext n
        exact (frontier_step p processed first n positive lower).symm
      simpa only [same] using updated
    · simpa [innerBody, Calls.loopContinuation, Calls.loopControl, Calls.loopControlStmt,
        Int.natCast_add, Int.natCast_one] using rest
  · have complete : workHeap base n (frontier p processed first) =
        workHeap base n (fun n => countWaysBound n (p :: processed)) := by
      unfold workHeap
      rw [table_congr (n + 1) (frontier p processed first)
        (fun n => countWaysBound n (p :: processed)) (fun n bounded => by
        simp [frontier, show n < first by omega])]
    refine ⟨env, locals, ?_⟩
    rw [← complete]
    exact BlockContinues.range_done (by simp [Calls.rangeContinues]; omega) BlockContinues.nil
termination_by n + 1 - first
decreasing_by omega

private theorem coin_pass (n p primesAddress : Nat) (positive : 0 < p)
    (processed : List Nat) (base : Mutable.Heap) (env : HeapCalls.Env)
    (locals : Locals env n primesAddress base.length)
    (getC : env "p" = some (number p)) :
    ∃ finalEnv, Locals finalEnv n primesAddress base.length ∧
      BlockContinues program env coinBody
        (workHeap base n (fun n => countWaysBound n processed)) finalEnv
        (workHeap base n (fun n => countWaysBound n (p :: processed))) := by
  obtain ⟨finalEnv, finalLocals, loop⟩ := inner_loop n p p primesAddress positive (by omega)
    processed base env locals getC
  have initial : frontier p processed p = fun n => countWaysBound n processed :=
    funext (fun n => frontier_start p processed n positive)
  rw [initial] at loop
  refine ⟨finalEnv, finalLocals, ?_⟩
  apply BlockContinues.for_range
    (.cons (Evaluates.lookup getC) (.cons
      (Evaluates.binary (av := number n) (bv := number 1) (value := number (n + 1))
        (middle := workHeap base n (fun n => countWaysBound n processed))
        (afterRight := workHeap base n (fun n => countWaysBound n processed))
        (finalHeap := workHeap base n (fun n => countWaysBound n processed))
        (Evaluates.lookup locals.targetValue) Evaluates.constant
        (by simp [Mutable.binary, scalarBinary, number])) .nil)) rfl
  exact loop

private theorem initialize_list (n : Nat) (env : HeapCalls.Env) (heap : Mutable.Heap)
    (getTarget : env "n" = some (number n)) :
    Evaluates program env (.binOp (.list [.constant (.int 0)]) .mult
      (.binOp (.name "n") .add (.constant (.int 1)))) heap (.ref (heap.length + 1))
      (workHeap (heap ++ [.list [number 0]]) n (fun _ => 0)) := by
  let base := heap ++ [.list [number 0]]
  apply Evaluates.binary (av := .ref heap.length) (bv := number (n + 1))
    (middle := base) (afterRight := base)
  · exact Evaluates.list (.cons Evaluates.constant .nil)
  · apply Evaluates.binary (Evaluates.lookup getTarget) Evaluates.constant
    simp [Mutable.binary, scalarBinary, number]
  · simp [Mutable.binary, show (BinOp.mult == BinOp.mult) = true from rfl,
      Mutable.array, Mutable.allocate, number, base, workHeap, table, Except.map]

private theorem initialize_zero (n : Nat) (base : Mutable.Heap) (env : HeapCalls.Env)
    (getDp : env "ways" = some (.ref base.length)) :
    BlockContinues program env
      [.assign [.subscript (.name "ways") (.constant (.int 0))] (.constant (.int 1))]
      (workHeap base n (fun _ => 0)) env
      (workHeap base n (fun n => countWaysBound n [])) := by
  have initial : (fun n : Nat => if n = 0 then 1 else 0) = fun n => countWaysBound n [] := by
    funext n
    exact (countWaysBound_nil n).symm
  rw [← initial]
  apply BlockContinues.setItem (by intro a b c; simp) Evaluates.constant
    (.cons (Evaluates.lookup getDp) (.cons Evaluates.constant .nil))
  rw [← workHeap_store base n (fun _ => 0) (fun n => if n = 0 then 1 else 0),
    ← table_set (n + 1) (fun _ => 0) 0 1 (by omega)]
  simp [Mutable.setItem, Mutable.index, workHeap, table, number, Except.map]


private abbrev outerBody : List Stmt :=
  .ifThen (.compare (.name "p") [(.gt, .name "n")]) [.breakLoop] [] :: coinBody

def active (n : Nat) (items : List Nat) : List Nat := items.takeWhile (fun p => decide (p ≤ n))

/-- The first oversized denomination ends traversal even if smaller values follow it. -/
private theorem outer_loop (items remaining processed : List Nat) (n position primesAddress : Nat)
    (positive : ∀ p ∈ remaining, 0 < p) (suffix : items.drop position = remaining)
    (base : Mutable.Heap) (found : base[primesAddress]? = some (.list (encode items)))
    (env : HeapCalls.Env) (locals : Locals env n primesAddress base.length) :
    ∃ finalEnv, Locals finalEnv n primesAddress base.length ∧
      BlockContinues program env [.sequenceStepLoop "p" (.ref primesAddress) position outerBody []]
        (workHeap base n (fun amount => countWaysBound amount processed)) finalEnv
        (workHeap base n (fun amount => countWaysBound amount ((active n remaining).reverse ++ processed))) := by
  have before : primesAddress < base.length := (List.getElem?_eq_some_iff.mp found).1
  have sequence (value : Nat → Nat) :
      Mutable.sequence (.ref primesAddress) (workHeap base n value) =
        some (.ok (encode items), workHeap base n value) := by
    simp [Mutable.sequence, workHeap, List.getElem?_append_left before, found]
  induction remaining generalizing processed position env with
  | nil =>
      have missing : items[position]? = none := by
        simpa [suffix] using (List.head?_drop (l := items) (i := position)).symm
      refine ⟨env, locals, ?_⟩
      simp only [active, List.takeWhile_nil, List.reverse_nil, List.nil_append]
      apply BlockContinues.sequence_done (source := .ref primesAddress)
        (items := encode items) (sequence (fun amount => countWaysBound amount processed))
        (by simp [encode, missing]) BlockContinues.nil
  | cons p tail ih =>
      have selected : items[position]? = some p := by
        simpa [suffix] using (List.head?_drop (l := items) (i := position)).symm
      let bound := setLocal env "p" (number p)
      have boundLocals : Locals bound n primesAddress base.length :=
        ⟨by simpa [bound, setLocal] using locals.targetValue,
         by simpa [bound, setLocal] using locals.primes,
         by simpa [bound, setLocal] using locals.ways⟩
      have tested : Evaluates program bound (.compare (.name "p") [(.gt, .name "n")])
          (workHeap base n (fun amount => countWaysBound amount processed))
          (.scalar (.bool (decide (n < p))))
          (workHeap base n (fun amount => countWaysBound amount processed)) := by
        apply Evaluates.compare (av := number p) (bv := number n)
          (Evaluates.lookup (by simp [bound, setLocal])) (Evaluates.lookup boundLocals.targetValue)
        simp [HeapCalls.compare, scalarCompare, evalCmpOp]
      by_cases tooLarge : n < p
      · refine ⟨bound, boundLocals, ?_⟩
        simp only [active, List.takeWhile_cons, show decide (p ≤ n) = false by simp [Nat.not_le.mpr tooLarge],
          Bool.false_eq_true, if_false, List.reverse_nil, List.nil_append]
        apply BlockContinues.sequence_step (source := .ref primesAddress) (items := encode items)
          (value := number p) (sequence (fun amount => countWaysBound amount processed)) (by simp [encode, selected])
        apply BlockContinues.branch_next (flag := true) (by simpa [tooLarge] using tested) rfl
        exact BlockContinues.break_loop (rest := []) rfl .nil
      · obtain ⟨middleEnv, middleLocals, pass⟩ := coin_pass n p primesAddress (positive p (by simp))
          processed base bound boundLocals (by simp [bound, setLocal])
        obtain ⟨finalEnv, finalLocals, rest⟩ := ih (p :: processed) (position + 1)
          (fun d member => positive d (by simp [member]))
          (by simp [List.drop_add_one_eq_tail_drop, suffix]) middleEnv middleLocals
        refine ⟨finalEnv, finalLocals, ?_⟩
        apply BlockContinues.sequence_step (source := .ref primesAddress) (items := encode items)
          (value := number p) (sequence (fun amount => countWaysBound amount processed)) (by simp [encode, selected])
        apply BlockContinues.branch_next (flag := false) (by simpa [tooLarge] using tested) rfl
        simpa [outerBody, coinBody, innerBody, Calls.loopContinuation, Calls.loopControl, Calls.loopControlStmt,
          active, show p ≤ n by omega, List.reverse_cons, List.append_assoc] using
          pass.append (BlockContinues.loop_resume rest)

/-- Arbitrary positive input lists retain their source order and their early stopping behavior. -/
theorem result (n : Nat) (items : List Nat) (positive : ∀ p ∈ items, 0 < p)
    (heap : Mutable.Heap) (address : Nat)
    (found : heap[address]? = some (.list (encode items))) :
    ReturnsValue program "count_prime_summations" [number n, .ref address] heap
      (number (countWaysBound n (active n items)))
      (heap ++ [.list [number 0], .list (table (n + 1) (fun amount => countWaysBound amount (active n items)))]) := by
  let base := heap ++ [.list [number 0]]
  let args := ((findFunctionIn program.body "count_prime_summations").getD ([], [])).1
  let body := ((findFunctionIn program.body "count_prime_summations").getD ([], [])).2
  let ready := setLocal (functionEnv args body [number n, .ref address] []) "ways" (.ref base.length)
  have locals : Locals ready n address base.length := ⟨rfl, rfl, rfl⟩
  have original : base[address]? = some (.list (encode items)) := by
    rw [List.getElem?_append_left (List.getElem?_eq_some_iff.mp found).1]
    exact found
  obtain ⟨finalEnv, finalLocals, loop⟩ := outer_loop items items [] n 0 address positive rfl base original ready locals
  have activePositive : ∀ p ∈ active n items, 0 < p :=
    fun p member => positive p (List.takeWhile_subset _ member)
  have row : (fun amount => countWaysBound amount ((active n items).reverse ++ [])) =
      fun amount => countWaysBound amount (active n items) := by
    funext amount
    simp only [List.append_nil, countWaysBound_reverse amount (active n items) activePositive]
  rw [row] at loop
  apply BlockReturns.function rfl rfl
  apply BlockReturns.assign (value := .ref base.length) (middle := workHeap base n (fun _ => 0))
  · simpa only [base, List.length_append, List.length_singleton] using
      initialize_list n (functionEnv args body [number n, .ref address] []) heap rfl
  · apply BlockContinues.returns (initialize_zero n base ready locals.ways)
    have scanned : BlockContinues program ready
        [.forIn (.name "p") (.name "primes") outerBody []]
        (workHeap base n (fun amount => countWaysBound amount [])) finalEnv
        (workHeap base n (fun amount => countWaysBound amount (active n items))) := by
      apply BlockContinues.for_sequence (items := encode items) (by intro expressions; simp) (Evaluates.lookup locals.primes)
      · have inside : address < base.length := (List.getElem?_eq_some_iff.mp original).1
        simp [Mutable.sequence, workHeap, List.getElem?_append_left inside, original]
      · exact loop
    apply scanned.returns
    apply BlockReturns.return_value
    apply Evaluates.subscript (by intro a b p; simp) (Evaluates.lookup finalLocals.ways)
      (Evaluates.lookup finalLocals.targetValue)
    have entry := table_lookup (n + 1) (fun amount => countWaysBound amount (active n items)) n (by omega)
    have retrieved := Mutable.getItem_list (workHeap base n (fun amount => countWaysBound amount (active n items)))
      base.length n (table (n + 1) (fun amount => countWaysBound amount (active n items)))
        (by simp [workHeap]) (by simp [table])
    simpa only [(List.getElem?_eq_some_iff.mp entry).2, workHeap, base, List.append_assoc] using retrieved

theorem active_of_bounded (n : Nat) (items : List Nat) (bounded : ∀ p ∈ items, p ≤ n) :
    active n items = items := by
  induction items with
  | nil => rfl
  | cons p items ih =>
      have head := bounded p (by simp)
      have tail := ih (fun q member => bounded q (by simp [member]))
      change (p :: items).takeWhile (fun q => decide (q ≤ n)) = p :: items
      rw [List.takeWhile_cons, if_pos (by simp [head])]
      exact congrArg (List.cons p) tail

/-- The sieve's prime list never triggers the break, so the helper counts every prime partition. -/
theorem primes_result (n : Nat) (heap : Mutable.Heap) (address : Nat)
    (found : heap[address]? = some (.list (encode (ProjectEulerStatements.PrimePartitions.primesUpTo n)))) :
    ReturnsValue program "count_prime_summations" [number n, .ref address] heap
      (number (ProjectEulerStatements.PrimePartitions.count n))
      (heap ++ [.list [number 0], .list (table (n + 1) (fun amount =>
        countWaysBound amount (ProjectEulerStatements.PrimePartitions.primesUpTo n)))]) := by
  have bounded : ∀ p ∈ ProjectEulerStatements.PrimePartitions.primesUpTo n, p ≤ n := by
    intro p member
    have within : p < n+1 := List.mem_range.mp (List.mem_filter.mp member).1
    omega
  simpa only [active_of_bounded n _ bounded, ProjectEulerStatements.PrimePartitions.count] using
    result n (ProjectEulerStatements.PrimePartitions.primesUpTo n)
      (ProjectEulerStatements.PrimePartitions.primesUpTo_positive n) heap address found

/-- The computed prime count leaves all preexisting objects unchanged. -/
theorem primes_append_result (n : Nat) (heap : Mutable.Heap) (address : Nat)
    (found : heap[address]? = some (.list (encode (ProjectEulerStatements.PrimePartitions.primesUpTo n)))) :
    ∃ extra, ReturnsValue program "count_prime_summations" [number n, .ref address] heap
      (number (ProjectEulerStatements.PrimePartitions.count n)) (heap ++ extra) :=
  ⟨_, primes_result n heap address found⟩

end count_prime_summations
end solvers.«77»
