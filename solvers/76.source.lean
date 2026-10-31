import «.formalsw».solvers.«76.py.ast»
import ProjectEulerStatements.P76
import lang.py.FormalSW.HeapCalls

/-! The ascending in-place partition table follows the positive-coin frontier. -/
namespace solvers.«76»
open FormalSW.PyAst FormalSW.PyAst.HeapCalls
open ProjectEulerStatements.P31 ProjectEulerStatements.CoinChange
abbrev program := Solvers.P76Ast.module
abbrev integer (n : Int) : Mutable.Value := .scalar (.int n)
abbrev number (n : Nat) : Mutable.Value := integer n
set_option Elab.async false

namespace solve
abbrev code : List Stmt := executableBody
  ((findFunctionIn program.body "solve").getD ([], [])).2

def table (size : Nat) (value : Nat → Nat) : List Mutable.Value :=
  (List.range size).map (fun n => number (value n))

theorem table_lookup (size : Nat) (value : Nat → Nat) (position : Nat)
    (inside : position < size) :
    (table size value)[position]? = some (number (value position)) := by
  simp [table, inside]

theorem table_set (size : Nat) (value : Nat → Nat) (position replacement : Nat)
    (inside : position < size) :
    (table size value).set position (number replacement) =
      table size (fun n => if n = position then replacement else value n) := by
  apply List.ext_getElem?
  intro n
  by_cases bounded : n < size
  · by_cases same : position = n
    · subst n
      rw [List.getElem?_set_self (by simpa [table] using inside),
        table_lookup size _ position inside]
      simp
    · rw [List.getElem?_set_ne same, table_lookup size value n bounded,
        table_lookup size _ n bounded]
      simp [Ne.symm same]
  · have outside : size ≤ n := by omega
    simp only [List.getElem?_eq_none (by simpa [table] using outside :
      ((table size value).set position (number replacement)).length ≤ n),
      List.getElem?_eq_none (by simpa [table] using outside :
        (table size (fun k => if k = position then replacement else value k)).length ≤ n)]

theorem table_congr (size : Nat) (f g : Nat → Nat) (equal : ∀ n, n < size → f n = g n) :
    table size f = table size g := by
  apply List.map_congr_left
  intro n member
  rw [equal n (List.mem_range.mp member)]

def workHeap (base : Mutable.Heap) (n : Nat) (value : Nat → Nat) : Mutable.Heap :=
  base ++ [.list (table (n + 1) value)]

theorem workHeap_store (base : Mutable.Heap) (n : Nat) (f g : Nat → Nat) :
    (workHeap base n f).set base.length (.list (table (n + 1) g)) = workHeap base n g := by
  simp [workHeap, List.set_append_right]

abbrev innerBody : List Stmt :=
  [.augAssign (.subscript (.name "dp") (.name "s")) .add
    (.subscript (.name "dp") (.binOp (.name "s") .sub (.name "part")))]
abbrev outerBody : List Stmt :=
  [.forIn (.name "s") (.call (.name "range")
    [.name "part", .binOp (.name "n") .add (.constant (.int 1))]) innerBody []]

structure Locals (env : HeapCalls.Env) (n dpAddress : Nat) : Prop where
  targetValue : env "n" = some (number n)
  dp : env "dp" = some (.ref dpAddress)

theorem update_cell (n part s : Nat) (inside : s ≤ n) (lower : part ≤ s)
    (value : Nat → Nat) (base : Mutable.Heap) (env : HeapCalls.Env)
    (getDp : env "dp" = some (.ref base.length))
    (getC : env "part" = some (number part)) (getS : env "s" = some (number s)) :
    BlockContinues program env innerBody (workHeap base n value) env
      (workHeap base n (fun n => if n = s then value s + value (s - part) else value n)) := by
  have found : (workHeap base n value)[base.length]? = some (.list (table (n + 1) value)) := by
    simp [workHeap]
  have readAt (query : Nat) (bound : query ≤ n) :
      Mutable.getItem (.ref base.length) (number query) (workHeap base n value) =
        some (.ok (number (value query)), workHeap base n value) := by
    have entry := table_lookup (n+1) value query (by omega)
    simpa only [(List.getElem?_eq_some_iff.mp entry).2] using
      Mutable.getItem_list (workHeap base n value) base.length query (table (n+1) value)
        found (by simpa [table] using Nat.lt_succ_of_le bound)
  apply BlockContinues.augItem (by intro a b part; simp)
    (Evaluates.lookup getDp) (Evaluates.lookup getS)
    (previous := number (value s)) (updated := number (value s + value (s - part)))
    (afterRead := workHeap base n value) (afterOperation := workHeap base n value)
  · exact readAt s inside
  · apply Evaluates.subscript (by intro a b part; simp) (Evaluates.lookup getDp)
      (Evaluates.binary (av := number s) (bv := number part) (value := number (s - part))
        (middle := workHeap base n value) (afterRight := workHeap base n value)
        (finalHeap := workHeap base n value) (Evaluates.lookup getS) (Evaluates.lookup getC)
        (by simp [Mutable.binary, scalarBinary, number, integer, Int.natCast_sub lower]))
    exact readAt (s - part) (by omega)
  · simp [inplace, Mutable.binary, scalarBinary, number, integer]
  · rw [← workHeap_store base n value
      (fun n => if n = s then value s + value (s - part) else value n),
      ← table_set (n + 1) value s (value s + value (s - part)) (by omega)]
    simp [Mutable.setItem, Mutable.index, found, table, number, integer, Except.map,
      show ¬ (s : Int) < 0 by omega, show ¬ n + 1 ≤ s by omega]

/-- During a coin pass, entries below the frontier already allow the new coin. -/
theorem inner_loop (n part first : Nat) (positive : 0 < part) (lower : part ≤ first)
    (processed : List Nat) (base : Mutable.Heap) (env : HeapCalls.Env)
    (locals : Locals env n base.length)
    (getC : env "part" = some (number part)) :
    ∃ finalEnv, Locals finalEnv n base.length ∧
      BlockContinues program env [.rangeStepLoop "s" first (n + 1) 1 innerBody []]
        (workHeap base n (frontier part processed first)) finalEnv
        (workHeap base n (fun n => countWaysBound n (part :: processed))) := by
  by_cases active : first ≤ n
  · let bound := setLocal env "s" (number first)
    have boundLocals : Locals bound n base.length :=
      ⟨by simpa [bound, setLocal] using locals.targetValue,
       by simpa [bound, setLocal] using locals.dp⟩
    have boundC : bound "part" = some (number part) := by simpa [bound, setLocal] using getC
    obtain ⟨finalEnv, finalLocals, rest⟩ := inner_loop n part (first + 1) positive (by omega)
      processed base bound boundLocals boundC
    refine ⟨finalEnv, finalLocals, ?_⟩
    apply BlockContinues.range_step (by simp [Calls.rangeContinues]; omega)
    apply BlockContinues.append (body := innerBody) (middleEnv := bound)
      (middleHeap := workHeap base n (frontier part processed (first + 1)))
    · have updated := update_cell n part first active lower
        (frontier part processed first) base bound
        boundLocals.dp boundC (by simp [bound, setLocal])
      have same : (fun n => if n = first then frontier part processed first first +
          frontier part processed first (first - part) else frontier part processed first n) =
          frontier part processed (first + 1) := by
        funext n
        exact (frontier_step part processed first n positive lower).symm
      simpa only [same] using updated
    · simpa [innerBody, Calls.loopContinuation, Calls.loopControl, Calls.loopControlStmt,
        Int.natCast_add, Int.natCast_one] using rest
  · have complete : workHeap base n (frontier part processed first) =
        workHeap base n (fun n => countWaysBound n (part :: processed)) := by
      unfold workHeap
      rw [table_congr (n + 1) (frontier part processed first)
        (fun n => countWaysBound n (part :: processed)) (fun n bounded => by
        simp [frontier, show n < first by omega])]
    refine ⟨env, locals, ?_⟩
    rw [← complete]
    exact BlockContinues.range_done (by simp [Calls.rangeContinues]; omega) BlockContinues.nil
termination_by n + 1 - first
decreasing_by omega

theorem coin_pass (n part : Nat) (positive : 0 < part)
    (processed : List Nat) (base : Mutable.Heap) (env : HeapCalls.Env)
    (locals : Locals env n base.length)
    (getC : env "part" = some (number part)) :
    ∃ finalEnv, Locals finalEnv n base.length ∧
      BlockContinues program env outerBody
        (workHeap base n (fun n => countWaysBound n processed)) finalEnv
        (workHeap base n (fun n => countWaysBound n (part :: processed))) := by
  obtain ⟨finalEnv, finalLocals, loop⟩ := inner_loop n part part positive (by omega)
    processed base env locals getC
  have initial : frontier part processed part = fun n => countWaysBound n processed :=
    funext (fun n => frontier_start part processed n positive)
  rw [initial] at loop
  refine ⟨finalEnv, finalLocals, ?_⟩
  apply BlockContinues.for_range
    (.cons (Evaluates.lookup getC) (.cons
      (Evaluates.binary (av := number n) (bv := number 1) (value := number (n + 1))
        (middle := workHeap base n (fun n => countWaysBound n processed))
        (afterRight := workHeap base n (fun n => countWaysBound n processed))
        (finalHeap := workHeap base n (fun n => countWaysBound n processed))
        (Evaluates.lookup locals.targetValue) Evaluates.constant
        (by simp [Mutable.binary, scalarBinary, number, integer])) .nil)) rfl
  exact loop

theorem initialize_list (n : Nat) (env : HeapCalls.Env) (heap : Mutable.Heap)
    (getTarget : env "n" = some (number n)) :
    Evaluates program env (.binOp (.list [.constant (.int 0)]) .mult
      (.binOp (.name "n") .add (.constant (.int 1)))) heap (.ref (heap.length + 1))
      (workHeap (heap ++ [.list [number 0]]) n (fun _ => 0)) := by
  let base := heap ++ [.list [number 0]]
  apply Evaluates.binary (av := .ref heap.length) (bv := number (n + 1))
    (middle := base) (afterRight := base)
  · exact Evaluates.list (.cons Evaluates.constant .nil)
  · apply Evaluates.binary (Evaluates.lookup getTarget) Evaluates.constant
    simp [Mutable.binary, scalarBinary, number, integer]
  · simp [Mutable.binary, show (BinOp.mult == BinOp.mult) = true from rfl,
      Mutable.array, Mutable.allocate, number, integer, base, workHeap, table, Except.map]

theorem initialize_zero (n : Nat) (base : Mutable.Heap) (env : HeapCalls.Env)
    (getDp : env "dp" = some (.ref base.length)) :
    BlockContinues program env
      [.assign [.subscript (.name "dp") (.constant (.int 0))] (.constant (.int 1))]
      (workHeap base n (fun _ => 0)) env
      (workHeap base n (fun n => countWaysBound n [])) := by
  have initial : (fun n : Nat => if n = 0 then 1 else 0) = fun n => countWaysBound n [] := by
    funext n
    exact (countWaysBound_nil n).symm
  rw [← initial]
  apply BlockContinues.setItem (by intro a b part; simp) Evaluates.constant
    (.cons (Evaluates.lookup getDp) (.cons Evaluates.constant .nil))
  rw [← workHeap_store base n (fun _ => 0) (fun n => if n = 0 then 1 else 0),
    ← table_set (n + 1) (fun _ => 0) 0 1 (by omega)]
  simp [Mutable.setItem, Mutable.index, workHeap, table, number, integer, Except.map]


/-- Each ascending denomination pass adds one new head to the recurrence's coin list. -/
theorem outer_loop (n first count : Nat) (processed : List Nat)
    (positive : 0 < first) (length : n-first=count)
    (base : Mutable.Heap) (env : HeapCalls.Env) (locals : Locals env n base.length) :
    ∃ finalEnv, Locals finalEnv n base.length ∧
      BlockContinues program env [.rangeStepLoop "part" first n 1 outerBody []]
        (workHeap base n (fun amount => countWaysBound amount processed)) finalEnv
        (workHeap base n (fun amount =>
          countWaysBound amount ((List.range' first count).reverse ++ processed))) := by
  induction count generalizing first processed env with
  | zero =>
      refine ⟨env, locals, ?_⟩
      simp only [List.range'_zero, List.reverse_nil, List.nil_append]
      exact BlockContinues.range_done (by simp [Calls.rangeContinues]; omega) .nil
  | succ count ih =>
      let entered := setLocal env "part" (number first)
      have enteredLocals : Locals entered n base.length :=
        ⟨by simpa [entered, setLocal] using locals.targetValue,
          by simpa [entered, setLocal] using locals.dp⟩
      obtain ⟨middleEnv, middleLocals, pass⟩ := coin_pass n first positive processed base entered
        enteredLocals (by simp [entered, setLocal])
      obtain ⟨finalEnv, finalLocals, rest⟩ := ih (first+1) (first :: processed)
        (by omega) (by omega) middleEnv middleLocals
      refine ⟨finalEnv, finalLocals, ?_⟩
      apply BlockContinues.range_step (by simp [Calls.rangeContinues]; omega)
      apply BlockContinues.append pass
      simpa [Calls.loopContinuation, Calls.loopControl, Calls.loopControlStmt,
        List.range'_succ, List.reverse_cons, List.append_assoc, Int.natCast_add] using rest

/-- Execute the original mutable DP table, including its zero entry and final indexed read. -/
theorem natural_result (n : Nat) (heap : Mutable.Heap) :
    ReturnsValue program "solve" [number n] heap
      (number (ProjectEulerStatements.P76.countPartitions n))
      (workHeap (heap ++ [.list [number 0]]) n
        (fun amount => countWaysBound amount (List.range' 1 (n-1)).reverse)) := by
  let base := heap ++ [.list [number 0]]
  let definition := (findFunctionIn program.body "solve").getD ([], [])
  let initial := functionEnv definition.1 definition.2 [number n] []
  let ready := setLocal initial "dp" (.ref base.length)
  have locals : Locals ready n base.length := ⟨rfl, rfl⟩
  obtain ⟨finalEnv, finalLocals, loop⟩ := outer_loop n 1 (n-1) [] (by decide) rfl
    base ready locals
  simp only [List.append_nil] at loop
  apply BlockReturns.function rfl rfl
  apply BlockReturns.branch (middle := heap) (flag := false) (value := .scalar (.bool false))
  · apply Evaluates.compare (av := number n) (bv := number 0)
      (Evaluates.lookup rfl) Evaluates.constant
    simp [HeapCalls.compare, scalarCompare, evalCmpOp]
  · rfl
  · apply BlockContinues.returns (BlockContinues.annAssign (value := .ref base.length)
      (finalHeap := workHeap base n (fun _ => 0)) ?_)
    · apply BlockContinues.returns (initialize_zero n base ready locals.dp)
      have scanned : BlockContinues program ready
          [.forIn (.name "part") (.call (.name "range") [.constant (.int 1), .name "n"])
            outerBody []] (workHeap base n (fun amount => countWaysBound amount [])) finalEnv
          (workHeap base n (fun amount => countWaysBound amount
            (List.range' 1 (n-1)).reverse)) := by
        apply BlockContinues.for_range
          (.cons Evaluates.constant (.cons (Evaluates.lookup locals.targetValue) .nil)) rfl
        exact loop
      apply scanned.returns
      apply BlockReturns.return_value
      apply Evaluates.subscript (by intros; intro h; cases h) (Evaluates.lookup finalLocals.dp)
        (Evaluates.lookup finalLocals.targetValue)
      have entry := table_lookup (n+1)
        (fun amount => countWaysBound amount (List.range' 1 (n-1)).reverse) n (by omega)
      have read := Mutable.getItem_list
        (workHeap base n (fun amount => countWaysBound amount (List.range' 1 (n-1)).reverse))
        base.length n
        (table (n+1) (fun amount => countWaysBound amount (List.range' 1 (n-1)).reverse))
        (by simp [workHeap]) (by simp [table])
      rw [(List.getElem?_eq_some_iff.mp entry).2] at read
      simpa only [← ProjectEulerStatements.P76.countWays_coins,
        ProjectEulerStatements.P76.countPartitions] using read
    · simpa only [base, List.length_append, List.length_singleton] using
        initialize_list n initial heap rfl

theorem result (n : Int) (heap : Mutable.Heap) :
    ∃ finalHeap, ReturnsValue program "solve" [integer n] heap
      (integer (ProjectEulerStatements.P76.result n)) finalHeap := by
  by_cases negative : n < 0
  · refine ⟨heap, ?_⟩
    rw [ProjectEulerStatements.P76.result_negative n negative]
    apply BlockReturns.function rfl rfl
    apply BlockReturns.branch (middle := heap) (flag := true) (value := .scalar (.bool true))
    · exact Evaluates.compare (Evaluates.lookup rfl) Evaluates.constant
        (by simp [HeapCalls.compare, scalarCompare, evalCmpOp, negative])
    · rfl
    · exact BlockReturns.return_value Evaluates.constant
  · lift n to Nat using (by omega : 0 ≤ n)
    refine ⟨workHeap (heap ++ [.list [number 0]]) n
      (fun amount => countWaysBound amount (List.range' 1 (n-1)).reverse), ?_⟩
    rw [ProjectEulerStatements.P76.result_nat]
    exact natural_result n heap

end solve
end solvers.«76»
