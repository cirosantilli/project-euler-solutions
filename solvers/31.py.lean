import «.formalsw».solvers.«31.py.ast»
import data.«project-euler-statements».data.py.project_euler_statements.«p31.py»
import lang.py.FormalSW.TruthCalls
import ProjectEulerStatements.Lib.CoinChange

namespace solvers.«31»
open FormalSW.PyAst FormalSW.PyAst.HeapCalls
open ProjectEulerStatements.P31 ProjectEulerStatements.CoinChange
private abbrev program := Solvers.P31Ast.module
private abbrev number (n : Nat) : Mutable.Value := .scalar (.int n)
private abbrev encode (items : List Nat) : List Mutable.Value := items.map number

set_option Elab.async false

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

private def workHeap (base : Mutable.Heap) (target : Nat) (value : Nat → Nat) : Mutable.Heap :=
  base ++ [.list (table (target + 1) value)]

private theorem workHeap_store (base : Mutable.Heap) (target : Nat) (f g : Nat → Nat) :
    (workHeap base target f).set base.length (.list (table (target + 1) g)) = workHeap base target g := by
  simp [workHeap, List.set_append_right]

private abbrev innerBody : List Stmt :=
  [.augAssign (.subscript (.name "dp") (.name "s")) .add
    (.subscript (.name "dp") (.binOp (.name "s") .sub (.name "c")))]
private abbrev outerBody : List Stmt :=
  [.forIn (.name "s") (.call (.name "range")
    [.name "c", .binOp (.name "target") .add (.constant (.int 1))]) innerBody []]

private structure Locals (env : HeapCalls.Env) (target coinsAddress dpAddress : Nat) : Prop where
  targetValue : env "target" = some (number target)
  coins : env "coins" = some (.ref coinsAddress)
  dp : env "dp" = some (.ref dpAddress)

private theorem update_cell (target c s : Nat) (inside : s ≤ target) (lower : c ≤ s)
    (value : Nat → Nat) (base : Mutable.Heap) (env : HeapCalls.Env)
    (getDp : env "dp" = some (.ref base.length))
    (getC : env "c" = some (number c)) (getS : env "s" = some (number s)) :
    BlockContinues program env innerBody (workHeap base target value) env
      (workHeap base target (fun n => if n = s then value s + value (s - c) else value n)) := by
  have found : (workHeap base target value)[base.length]? = some (.list (table (target + 1) value)) := by
    simp [workHeap]
  have readAt (n : Nat) (bound : n ≤ target) :
      Mutable.getItem (.ref base.length) (number n) (workHeap base target value) =
        some (.ok (number (value n)), workHeap base target value) := by
    have entry := table_lookup (target + 1) value n (by omega)
    simpa only [(List.getElem?_eq_some_iff.mp entry).2] using
      Mutable.getItem_list (workHeap base target value) base.length n _ found (by simpa [table] using Nat.lt_succ_of_le bound)
  apply BlockContinues.augItem (by intro a b c; simp) (Evaluates.lookup getDp) (Evaluates.lookup getS)
    (previous := number (value s)) (updated := number (value s + value (s - c)))
    (afterRead := workHeap base target value) (afterOperation := workHeap base target value)
  · exact readAt s inside
  · apply Evaluates.subscript (by intro a b c; simp) (Evaluates.lookup getDp)
      (Evaluates.binary (av := number s) (bv := number c) (value := number (s - c))
        (middle := workHeap base target value) (afterRight := workHeap base target value)
        (finalHeap := workHeap base target value) (Evaluates.lookup getS) (Evaluates.lookup getC)
        (by simp [Mutable.binary, scalarBinary, number, Int.natCast_sub lower]))
    exact readAt (s - c) (by omega)
  · simp [inplace, Mutable.binary, scalarBinary, number]
  · rw [← workHeap_store base target value
      (fun n => if n = s then value s + value (s - c) else value n),
      ← table_set (target + 1) value s (value s + value (s - c)) (by omega)]
    simp [Mutable.setItem, Mutable.index, found, table, number, Except.map,
      show ¬ (s : Int) < 0 by omega, show ¬ target + 1 ≤ s by omega]

/-- During a coin pass, entries below the frontier already allow the new coin. -/
private theorem inner_loop (target c first coinsAddress : Nat) (positive : 0 < c) (lower : c ≤ first)
    (processed : List Nat) (base : Mutable.Heap) (env : HeapCalls.Env)
    (locals : Locals env target coinsAddress base.length)
    (getC : env "c" = some (number c)) :
    ∃ finalEnv, Locals finalEnv target coinsAddress base.length ∧
      BlockContinues program env [.rangeStepLoop "s" first (target + 1) 1 innerBody []]
        (workHeap base target (frontier c processed first)) finalEnv
        (workHeap base target (fun n => countWaysBound n (c :: processed))) := by
  by_cases active : first ≤ target
  · let bound := setLocal env "s" (number first)
    have boundLocals : Locals bound target coinsAddress base.length :=
      ⟨by simpa [bound, setLocal] using locals.targetValue,
       by simpa [bound, setLocal] using locals.coins,
       by simpa [bound, setLocal] using locals.dp⟩
    have boundC : bound "c" = some (number c) := by simpa [bound, setLocal] using getC
    obtain ⟨finalEnv, finalLocals, rest⟩ := inner_loop target c (first + 1) coinsAddress positive (by omega)
      processed base bound boundLocals boundC
    refine ⟨finalEnv, finalLocals, ?_⟩
    apply BlockContinues.range_step (by simp [Calls.rangeContinues]; omega)
    apply BlockContinues.append (body := innerBody) (middleEnv := bound)
      (middleHeap := workHeap base target (frontier c processed (first + 1)))
    · have updated := update_cell target c first active lower (frontier c processed first) base bound
        boundLocals.dp boundC (by simp [bound, setLocal])
      have same : (fun n => if n = first then frontier c processed first first +
          frontier c processed first (first - c) else frontier c processed first n) =
          frontier c processed (first + 1) := by
        funext n
        exact (frontier_step c processed first n positive lower).symm
      simpa only [same] using updated
    · simpa [innerBody, Calls.loopContinuation, Calls.loopControl, Calls.loopControlStmt,
        Int.natCast_add, Int.natCast_one] using rest
  · have complete : workHeap base target (frontier c processed first) =
        workHeap base target (fun n => countWaysBound n (c :: processed)) := by
      unfold workHeap
      rw [table_congr (target + 1) (frontier c processed first)
        (fun n => countWaysBound n (c :: processed)) (fun n bounded => by
        simp [frontier, show n < first by omega])]
    refine ⟨env, locals, ?_⟩
    rw [← complete]
    exact BlockContinues.range_done (by simp [Calls.rangeContinues]; omega) BlockContinues.nil
termination_by target + 1 - first
decreasing_by omega

private theorem coin_pass (target c coinsAddress : Nat) (positive : 0 < c)
    (processed : List Nat) (base : Mutable.Heap) (env : HeapCalls.Env)
    (locals : Locals env target coinsAddress base.length)
    (getC : env "c" = some (number c)) :
    ∃ finalEnv, Locals finalEnv target coinsAddress base.length ∧
      BlockContinues program env outerBody
        (workHeap base target (fun n => countWaysBound n processed)) finalEnv
        (workHeap base target (fun n => countWaysBound n (c :: processed))) := by
  obtain ⟨finalEnv, finalLocals, loop⟩ := inner_loop target c c coinsAddress positive (by omega)
    processed base env locals getC
  have initial : frontier c processed c = fun n => countWaysBound n processed :=
    funext (fun n => frontier_start c processed n positive)
  rw [initial] at loop
  refine ⟨finalEnv, finalLocals, ?_⟩
  apply BlockContinues.for_range
    (.cons (Evaluates.lookup getC) (.cons
      (Evaluates.binary (av := number target) (bv := number 1) (value := number (target + 1))
        (middle := workHeap base target (fun n => countWaysBound n processed))
        (afterRight := workHeap base target (fun n => countWaysBound n processed))
        (finalHeap := workHeap base target (fun n => countWaysBound n processed))
        (Evaluates.lookup locals.targetValue) Evaluates.constant
        (by simp [Mutable.binary, scalarBinary, number])) .nil)) rfl
  exact loop

/-- Processed coins are stored in reverse order, matching the recurrence's new head. -/
private theorem outer_loop (items remaining processed : List Nat) (target position coinsAddress : Nat)
    (positive : ∀ c ∈ remaining, 0 < c) (suffix : items.drop position = remaining)
    (base : Mutable.Heap) (found : base[coinsAddress]? = some (.list (encode items)))
    (env : HeapCalls.Env) (locals : Locals env target coinsAddress base.length) :
    ∃ finalEnv, Locals finalEnv target coinsAddress base.length ∧
      BlockContinues program env [.sequenceStepLoop "c" (.ref coinsAddress) position outerBody []]
        (workHeap base target (fun n => countWaysBound n processed)) finalEnv
        (workHeap base target (fun n => countWaysBound n (remaining.reverse ++ processed))) := by
  have before : coinsAddress < base.length := (List.getElem?_eq_some_iff.mp found).1
  have sequence (value : Nat → Nat) :
      Mutable.sequence (.ref coinsAddress) (workHeap base target value) =
        some (.ok (encode items), workHeap base target value) := by
    simp [Mutable.sequence, workHeap, List.getElem?_append_left before, found]
  induction remaining generalizing processed position env with
  | nil =>
      have missing : items[position]? = none := by
        simpa [suffix] using (List.head?_drop (l := items) (i := position)).symm
      refine ⟨env, locals, ?_⟩
      simp only [List.reverse_nil, List.nil_append]
      apply BlockContinues.sequence_done (source := .ref coinsAddress)
        (items := encode items) (sequence (fun n => countWaysBound n processed))
        (by simp [encode, missing]) BlockContinues.nil
  | cons c tail ih =>
      have selected : items[position]? = some c := by
        simpa [suffix] using (List.head?_drop (l := items) (i := position)).symm
      let bound := setLocal env "c" (number c)
      have boundLocals : Locals bound target coinsAddress base.length :=
        ⟨by simpa [bound, setLocal] using locals.targetValue,
         by simpa [bound, setLocal] using locals.coins,
         by simpa [bound, setLocal] using locals.dp⟩
      obtain ⟨middleEnv, middleLocals, pass⟩ := coin_pass target c coinsAddress (positive c (by simp))
        processed base bound boundLocals (by simp [bound, setLocal])
      obtain ⟨finalEnv, finalLocals, rest⟩ := ih (c :: processed) (position + 1) (fun d member => positive d (by simp [member])) (by simp [List.drop_add_one_eq_tail_drop, suffix]) middleEnv middleLocals
      refine ⟨finalEnv, finalLocals, ?_⟩
      apply BlockContinues.sequence_step (source := .ref coinsAddress) (items := encode items)
        (value := number c) (sequence (fun n => countWaysBound n processed)) (by simp [encode, selected])
      apply BlockContinues.append pass
      simpa [outerBody, innerBody, Calls.loopContinuation, Calls.loopControl, Calls.loopControlStmt,
        List.reverse_cons, List.append_assoc] using rest

private theorem initialize_list (target : Nat) (env : HeapCalls.Env) (heap : Mutable.Heap)
    (getTarget : env "target" = some (number target)) :
    Evaluates program env (.binOp (.list [.constant (.int 0)]) .mult
      (.binOp (.name "target") .add (.constant (.int 1)))) heap (.ref (heap.length + 1))
      (workHeap (heap ++ [.list [number 0]]) target (fun _ => 0)) := by
  let base := heap ++ [.list [number 0]]
  apply Evaluates.binary (av := .ref heap.length) (bv := number (target + 1))
    (middle := base) (afterRight := base)
  · exact Evaluates.list (.cons Evaluates.constant .nil)
  · apply Evaluates.binary (Evaluates.lookup getTarget) Evaluates.constant
    simp [Mutable.binary, scalarBinary, number]
  · simp [Mutable.binary, show (BinOp.mult == BinOp.mult) = true from rfl,
      Mutable.array, Mutable.allocate, number, base, workHeap, table, Except.map]

private theorem initialize_zero (target : Nat) (base : Mutable.Heap) (env : HeapCalls.Env)
    (getDp : env "dp" = some (.ref base.length)) :
    BlockContinues program env
      [.assign [.subscript (.name "dp") (.constant (.int 0))] (.constant (.int 1))]
      (workHeap base target (fun _ => 0)) env
      (workHeap base target (fun n => countWaysBound n [])) := by
  have initial : (fun n : Nat => if n = 0 then 1 else 0) = fun n => countWaysBound n [] := by
    funext n
    exact (countWaysBound_nil n).symm
  rw [← initial]
  apply BlockContinues.setItem (by intro a b c; simp) Evaluates.constant
    (.cons (Evaluates.lookup getDp) (.cons Evaluates.constant .nil))
  rw [← workHeap_store base target (fun _ => 0) (fun n => if n = 0 then 1 else 0),
    ← table_set (target + 1) (fun _ => 0) 0 1 (by omega)]
  simp [Mutable.setItem, Mutable.index, workHeap, table, number, Except.map]

/-- The source DP works for arbitrary positive denominations, including repeated or unsorted ones. -/
theorem solve.result (target : Nat) (items : List Nat) (positive : ∀ c ∈ items, 0 < c)
    (heap : Mutable.Heap) (address : Nat)
    (found : heap[address]? = some (.list (encode items))) :
    ReturnsValue program "solve" [number target, .ref address] heap
      (number (countWaysBound target items))
      (heap ++ [.list [number 0], .list (table (target + 1) (fun n => countWaysBound n items))]) := by
  let base := heap ++ [.list [number 0]]
  let args := ((findFunctionIn program.body "solve").getD ([], [])).1
  let body := ((findFunctionIn program.body "solve").getD ([], [])).2
  let ready := setLocal (functionEnv args body [number target, .ref address] []) "dp" (.ref base.length)
  have locals : Locals ready target address base.length := ⟨rfl, rfl, rfl⟩
  have original : base[address]? = some (.list (encode items)) := by
    rw [List.getElem?_append_left (List.getElem?_eq_some_iff.mp found).1]
    exact found
  obtain ⟨finalEnv, finalLocals, loop⟩ := outer_loop items items [] target 0 address positive rfl base original ready locals
  have row : (fun n => countWaysBound n (items.reverse ++ [])) = fun n => countWaysBound n items := by
    funext n
    simp only [List.append_nil, countWaysBound_reverse n items positive]
  rw [row] at loop
  apply BlockReturns.function rfl rfl
  apply BlockReturns.assign (value := .ref base.length)
    (middle := workHeap base target (fun _ => 0))
  · simpa only [base, List.length_append, List.length_singleton] using
      initialize_list target (functionEnv args body [number target, .ref address] []) heap rfl
  · apply BlockContinues.returns (initialize_zero target base ready locals.dp)
    have scanned : BlockContinues program ready
        [.forIn (.name "c") (.name "coins") outerBody []]
        (workHeap base target (fun n => countWaysBound n [])) finalEnv
        (workHeap base target (fun n => countWaysBound n items)) := by
      apply BlockContinues.for_sequence (items := encode items) (by intro expressions; simp) (Evaluates.lookup locals.coins)
      · have inside : address < base.length := (List.getElem?_eq_some_iff.mp original).1
        simp [Mutable.sequence, workHeap, List.getElem?_append_left inside, original]
      · exact loop
    apply scanned.returns
    apply BlockReturns.return_value
    apply Evaluates.subscript (by intro a b c; simp) (Evaluates.lookup finalLocals.dp)
      (Evaluates.lookup finalLocals.targetValue)
    have entry := table_lookup (target + 1) (fun n => countWaysBound n items) target (by omega)
    have retrieved := Mutable.getItem_list (workHeap base target (fun n => countWaysBound n items))
      base.length target (table (target + 1) (fun n => countWaysBound n items))
        (by simp [workHeap]) (by simp [table])
    simpa only [(List.getElem?_eq_some_iff.mp entry).2, workHeap, base, List.append_assoc] using retrieved

/-- Both implementations preserve the caller's original heap and return the same scalar. -/
theorem solve.equiv_naturals (target : Nat) (items : List Nat)
    (positive : ∀ c ∈ items, 0 < c) (heap : Mutable.Heap) (address : Nat)
    (valid : heap.Valid) (found : heap[address]? = some (.list (encode items))) :
    (HeapCalls.call program "solve" [number target, .ref address] heap).Equiv
      (HeapCalls.call P31Ast.module "count_ways_bound" [number target, .ref address] heap) := by
  have optimized := solve.result target items positive heap address found
  obtain ⟨extra, reference⟩ :=
    data.«project-euler-statements».data.py.project_euler_statements.p31.count_ways_bound.result
      target items positive heap address found
  exact Mutable.Call.equiv_of_finishes optimized.finishes reference.finishes
    (Mutable.Completion.equivalent_allocations heap
      [.list [number 0], .list (table (target + 1) (fun n => countWaysBound n items))]
      extra (number (countWaysBound target items)) [] valid trivial (by simp))

private abbrev inputEnv (target : Int) (address : Nat) : HeapCalls.Env :=
  fun name => if name = "target" then some (.scalar (.int target))
    else if name = "coins" then some (.ref address) else none

private theorem condition_result (target : Int) (items : List Int) (heap : Mutable.Heap) (address : Nat)
    (found : heap[address]? = some (.list (items.map (fun n => .scalar (.int n))))) :
    Evaluates program (inputEnv target address) Solvers.P31Ast.requires_solve_0 heap
      (.scalar (.bool (decide (0 ≤ target ∧ ∀ c ∈ items, 0 < c)))) heap := by
  have first : Evaluates program (inputEnv target address)
      (.compare (.name "target") [(.gtE, .constant (.int 0))]) heap
      (.scalar (.bool (decide (0 ≤ target)))) heap := by
    exact Evaluates.compare (Evaluates.lookup rfl) Evaluates.constant
      (by simp [HeapCalls.compare, scalarCompare, evalCmpOp])
  by_cases nonnegative : 0 ≤ target
  · simp only [nonnegative, true_and]
    apply Evaluates.boolean_next (flag := true) (by simpa [nonnegative] using first) rfl rfl
    apply Evaluates.boolean_last
      (value := .scalar (.bool (decide (∀ c ∈ items, 0 < c))))
    · exact (Evaluates.all_positive (name := "c") (sourceName := "coins") (values := items)
          (module := program) (env := inputEnv target address) (heap := heap) (middle := heap)
          (Evaluates.lookup rfl) (by simp [Mutable.sequence, found]))
    · rfl
  · have rejected : decide (0 ≤ target ∧ ∀ c ∈ items, 0 < c) = false := by simp [nonnegative]
    rw [rejected]
    exact Evaluates.boolean_stop (flag := false) (by simpa [nonnegative] using first) rfl rfl

theorem requires_solve_0_supported (heap : Mutable.Heap) (target : Int) (address : Nat)
    (_valid : heap.Valid) (typed : Mutable.Value.HasType heap (.ref address) "list[int]") :
    (HeapCalls.evaluate program (inputEnv target address) Solvers.P31Ast.requires_solve_0 heap).Supported := by
  obtain ⟨items, found⟩ := Mutable.Value.ints_of_hasType typed
  exact Execution.supported_of_finishes (condition_result target items heap address found).finishes

private theorem domain (heap : Mutable.Heap) (target : Int) (address : Nat)
    (typed : Mutable.Value.HasType heap (.ref address) "list[int]")
    (requires : HeapCalls.Condition program (inputEnv target address) Solvers.P31Ast.requires_solve_0 heap) :
    ∃ (amount : Nat) (items : List Nat), target = (amount : Int) ∧
      (∀ c ∈ items, 0 < c) ∧ heap[address]? = some (.list (encode items)) := by
  obtain ⟨integers, found⟩ := Mutable.Value.ints_of_hasType typed
  have accepted : 0 ≤ target ∧ ∀ c ∈ integers, 0 < c := by
    simpa using (condition_result target integers heap address found).condition_iff.mp requires
  refine ⟨target.toNat, integers.map Int.toNat, (Int.toNat_of_nonneg accepted.1).symm, ?_, ?_⟩
  · intro c member
    obtain ⟨integer, present, rfl⟩ := List.mem_map.mp member
    have positive := accepted.2 integer present
    omega
  · have encoded : encode (integers.map Int.toNat) = integers.map (fun n => .scalar (.int n)) := by
      simp only [encode, List.map_map]
      apply List.map_congr_left
      intro n member
      have nonnegative : 0 ≤ n := by
        have positive := accepted.2 n member
        omega
      simp only [Function.comp_apply, number, Int.toNat_of_nonneg nonnegative]
    simpa only [encoded] using found

theorem solve.«Equiv(count_ways_bound)» (heap : Mutable.Heap) (target : Int) (address : Nat)
    (valid : heap.Valid) (typed : Mutable.Value.HasType heap (.ref address) "list[int]")
    (requires : HeapCalls.Condition program (inputEnv target address) Solvers.P31Ast.requires_solve_0 heap) :
    (HeapCalls.call program "solve" [.scalar (.int target), .ref address] heap).Equiv
      (HeapCalls.call P31Ast.module "count_ways_bound" [.scalar (.int target), .ref address] heap) := by
  obtain ⟨amount, items, rfl, positive, found⟩ := domain heap target address typed requires
  exact solve.equiv_naturals amount items positive heap address valid found

theorem solve.«Supported» :
    (∀ (heap : Mutable.Heap) (target : Int) (address : Nat), heap.Valid →
      Mutable.Value.HasType heap (.ref address) "list[int]" →
      HeapCalls.Condition program (inputEnv target address) Solvers.P31Ast.requires_solve_0 heap →
      (HeapCalls.call program "solve" [.scalar (.int target), .ref address] heap).Supported) ∧
    (∀ (heap : Mutable.Heap) (target : Int) (address : Nat), heap.Valid →
      Mutable.Value.HasType heap (.ref address) "list[int]" →
      HeapCalls.Condition program (inputEnv target address) Solvers.P31Ast.requires_solve_0 heap →
      (HeapCalls.call P31Ast.module "count_ways_bound" [.scalar (.int target), .ref address] heap).Supported) := by
  constructor
  · intro heap target address _valid typed requires
    obtain ⟨amount, items, rfl, positive, found⟩ := domain heap target address typed requires
    exact (solve.result amount items positive heap address found).returns.supported
  · intro heap target address _valid typed requires
    obtain ⟨amount, items, rfl, positive, found⟩ := domain heap target address typed requires
    obtain ⟨_, reference⟩ :=
      data.«project-euler-statements».data.py.project_euler_statements.p31.count_ways_bound.result
        amount items positive heap address found
    exact reference.returns.supported

end solvers.«31»
