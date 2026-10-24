import «.formalsw».solvers.«21.py.ast»
import data.«project-euler-statements».data.py.project_euler_statements.«p21.py»
import FormalSW
import ProjectEulerStatements.Lib.DivisorSieve
import ProjectEulerStatements.Lib.DivisorSumFactors

namespace solvers.«21»
open FormalSW.PyAst FormalSW.PyAst.HeapCalls
private abbrev program := Solvers.P21Ast.module

/-- The factorization helper handles small and negative inputs before dividing. -/
theorem factorization.small_result (n : Int) (small : n ≤ 1) (heap : Mutable.Heap) :
    ReturnsValue program "proper_divisor_sum_factorization" [.scalar (.int n)] heap (.scalar (.int 0)) heap := by
  apply BlockReturns.function rfl rfl
  apply BlockReturns.branch (value := .scalar (.bool true)) (flag := true) (middle := heap)
  · apply Evaluates.compare (Evaluates.lookup rfl) Evaluates.constant
    simp [HeapCalls.compare, scalarCompare, evalCmpOp, small]
  · rfl
  · exact BlockReturns.return_value Evaluates.constant

namespace Sieve
open ProjectEulerStatements.DivisorSieve

private abbrev number (n : Nat) : Mutable.Value := .scalar (.int n)

/-- A source list whose entries are the current mathematical sieve values. -/
def table (size : Nat) (value : Nat → Nat) : List Mutable.Value :=
  (List.range size).map (fun n => number (value n))

@[simp] theorem table_length (size : Nat) (value : Nat → Nat) : (table size value).length = size := by
  simp [table]

theorem table_lookup (size : Nat) (value : Nat → Nat) (position : Nat) (inside : position < size) :
    (table size value)[position]? = some (number (value position)) := by
  simp [table, inside]

theorem table_set (size : Nat) (value : Nat → Nat) (position replacement : Nat) (inside : position < size) :
    (table size value).set position (number replacement) =
      table size (fun n => if n = position then replacement else value n) := by
  apply List.ext_getElem?
  intro n
  by_cases bounded : n < size
  · by_cases same : position = n
    · subst n
      rw [List.getElem?_set_self (by simpa using inside), table_lookup size _ position inside]
      simp
    · rw [List.getElem?_set_ne same, table_lookup size value n bounded,
        table_lookup size _ n bounded]
      simp [Ne.symm same]
  · have outside : size ≤ n := by omega
    simp only [List.getElem?_eq_none (by simpa using outside : ((table size value).set position (number replacement)).length ≤ n),
      List.getElem?_eq_none (by simpa using outside : (table size (fun k => if k = position then replacement else value k)).length ≤ n)]

private def definition : List Arg × List Stmt :=
  (findFunctionIn program.body "proper_divisor_sums_sieve").getD ([], [])

private def outerBody : List Stmt :=
  match ((executableBody definition.2)[1]? : Option Stmt) with
  | some (.forIn _ _ body _) => body
  | _ => []

private def innerBody : List Stmt :=
  match (outerBody[0]? : Option Stmt) with
  | some (.forIn _ _ body _) => body
  | _ => []

private structure Locals (env : Mutable.Env) (maximum address : Nat) : Prop where
  bound : env "n_max" = some (number maximum)
  array : env "s" = some (.ref address)

/-- The source augmented subscription performs one exact in-place table update. -/
private theorem increment (size i j address : Nat) (inside : j < size) (value : Nat → Nat)
    (env : Mutable.Env) (heap : Mutable.Heap)
    (array : env "s" = some (.ref address)) (divisor : env "i" = some (number i))
    (index : env "j" = some (number j))
    (found : heap[address]? = some (.list (table size value))) :
    BlockContinues program env innerBody heap env
      (heap.set address (.list (table size (fun n => if n = j then value j + i else value n)))) := by
  apply BlockContinues.augItem (by intro a b c; simp) (Evaluates.lookup array) (Evaluates.lookup index)
    (previous := number (value j)) (updated := number (value j + i))
    (afterRead := heap) (afterOperation := heap) (rhs := Evaluates.lookup divisor)
  · have item := table_lookup size value j inside
    have bound : j < (table size value).length := by simpa using inside
    simpa only [(List.getElem?_eq_some_iff.mp item).2] using Mutable.getItem_list heap address j _ found bound
  · simp [inplace, Mutable.binary, scalarBinary, number]
  · have nonnegative : ¬ (j : Int) < 0 := by omega
    have bounded : ¬ size ≤ j := by omega
    rw [← table_set size value j (value j + i) inside]
    simp [Mutable.setItem, Mutable.index, found, nonnegative, bounded, number, Except.map]

theorem table_congr (size : Nat) (f g : Nat → Nat) (equal : ∀ n, n < size → f n = g n) :
    table size f = table size g := by
  apply List.map_congr_left
  intro n member
  rw [equal n (List.mem_range.mp member)]

private def arrayHeap (base : Mutable.Heap) (size : Nat) (value : Nat → Nat) : Mutable.Heap :=
  base ++ [.list (table size value)]

private theorem arrayHeap_store (base : Mutable.Heap) (size : Nat) (f g : Nat → Nat) :
    (arrayHeap base size f).set base.length (.list (table size g)) = arrayHeap base size g := by
  simp [arrayHeap, List.set_append_right]

/-- The inner source loop visits every remaining multiple once, with a positive stride. -/
private theorem inner_loop (maximum i j : Nat) (positive : 0 < i)
    (lower : 2 * i ≤ j) (multiple : i ∣ j) (env : Mutable.Env) (base : Mutable.Heap)
    (locals : Locals env maximum base.length) (divisor : env "i" = some (number i)) :
    ∃ finalEnv, Locals finalEnv maximum base.length ∧ finalEnv "i" = some (number i) ∧
      BlockContinues program env [.rangeStepLoop "j" j (maximum + 1) i innerBody []]
        (arrayHeap base (maximum + 1) (row (sumsBefore i) i j)) finalEnv
        (arrayHeap base (maximum + 1) (sumsBefore (i + 1))) := by
  by_cases active : j < maximum + 1
  · let nextEnv := setLocal env "j" (number j)
    have nextLocals : Locals nextEnv maximum base.length := by
      exact ⟨by simpa [nextEnv, setLocal] using locals.bound, by simpa [nextEnv, setLocal] using locals.array⟩
    have nextDivisor : nextEnv "i" = some (number i) := by simpa [nextEnv, setLocal] using divisor
    obtain ⟨finalEnv, finalLocals, finalDivisor, recurse⟩ := inner_loop maximum i (j + i) positive (by omega)
      (dvd_add multiple (dvd_refl i)) nextEnv base nextLocals nextDivisor
    refine ⟨finalEnv, finalLocals, finalDivisor, ?_⟩
    apply BlockContinues.range_step (by simp [Calls.rangeContinues]; omega)
    apply BlockContinues.append (body := innerBody) (middleEnv := nextEnv)
      (middleHeap := arrayHeap base (maximum + 1) (row (sumsBefore i) i (j + i)))
    · have updated := increment (maximum + 1) i j base.length active (row (sumsBefore i) i j)
        nextEnv (arrayHeap base (maximum + 1) (row (sumsBefore i) i j)) nextLocals.array nextDivisor
        (by simp [nextEnv, setLocal]) (by simp [arrayHeap])
      have values : (fun n => if n = j then row (sumsBefore i) i j j + i else row (sumsBefore i) i j n) =
          row (sumsBefore i) i (j + i) := by
        funext n
        by_cases same : n = j
        · subst n
          simpa using (row_step (sumsBefore i) i j j positive lower multiple).symm
        · simpa only [if_neg same] using (row_step (sumsBefore i) i j n positive lower multiple).symm
      simpa only [arrayHeap_store, values] using updated
    · simpa [Calls.loopContinuation, innerBody] using recurse
  · have complete : arrayHeap base (maximum + 1) (row (sumsBefore i) i j) =
        arrayHeap base (maximum + 1) (sumsBefore (i + 1)) := by
      exact congrArg (fun items => base ++ [Mutable.Object.list items])
        (table_congr (maximum + 1) _ _ (fun n bound => row_complete i j n positive (by omega)))
    refine ⟨env, locals, divisor, ?_⟩
    rw [← complete]
    exact BlockContinues.range_done (by simp [Calls.rangeContinues, positive]; omega) BlockContinues.nil
termination_by maximum + 1 - j
decreasing_by omega

/-- Run the actual nested range statement for one positive divisor. -/
private theorem divisor_pass (maximum i : Nat) (positive : 0 < i) (env : Mutable.Env) (base : Mutable.Heap)
    (locals : Locals env maximum base.length) (divisor : env "i" = some (number i)) :
    ∃ finalEnv, Locals finalEnv maximum base.length ∧
      BlockContinues program env outerBody (arrayHeap base (maximum + 1) (sumsBefore i)) finalEnv
        (arrayHeap base (maximum + 1) (sumsBefore (i + 1))) := by
  obtain ⟨finalEnv, finalLocals, _, scanned⟩ := inner_loop maximum i (2 * i) positive (by omega)
    (dvd_mul_left i 2) env base locals divisor
  have initial : row (sumsBefore i) i (2 * i) = sumsBefore i := funext (row_start _ i)
  rw [initial] at scanned
  refine ⟨finalEnv, finalLocals, ?_⟩
  apply BlockContinues.for_range
    (values := [number (2 * i), number (maximum + 1), number i])
    (start := 2 * i) (stop := maximum + 1) (stride := i)
    (middle := arrayHeap base (maximum + 1) (sumsBefore i))
  · apply Evaluates.Arguments.cons (middle := arrayHeap base (maximum + 1) (sumsBefore i))
    · apply Evaluates.binary (Evaluates.lookup divisor) Evaluates.constant
      simp [Mutable.binary, scalarBinary, number, Int.mul_comm]
    · apply Evaluates.Arguments.cons (middle := arrayHeap base (maximum + 1) (sumsBefore i))
      · apply Evaluates.binary (Evaluates.lookup locals.bound) Evaluates.constant
        simp [Mutable.binary, scalarBinary, number]
      · exact .cons (Evaluates.lookup divisor) .nil
  · change (if (i : Int) = 0 then .error { typeName := "ValueError", args := [] }
        else .ok (↑(2 * i), ↑(maximum + 1), ↑i) : Except ExceptionObject (Int × Int × Int)) = _
    simp [positive.ne']
  · simpa only [number, Nat.cast_mul, Nat.cast_ofNat] using scanned

/-- The outer source loop grows the prefix of processed divisors. -/
private theorem outer_loop (count start maximum : Nat) (positive : 0 < start)
    (stop : start + count = maximum / 2 + 1) (env : Mutable.Env) (base : Mutable.Heap)
    (locals : Locals env maximum base.length) :
    ∃ finalEnv, Locals finalEnv maximum base.length ∧
      BlockContinues program env [.rangeStepLoop "i" start (maximum / 2 + 1) 1 outerBody []]
        (arrayHeap base (maximum + 1) (sumsBefore start)) finalEnv
        (arrayHeap base (maximum + 1) (sumsBefore (maximum / 2 + 1))) := by
  induction count generalizing start env with
  | zero =>
      have same : start = maximum / 2 + 1 := by omega
      refine ⟨env, locals, ?_⟩
      rw [same]
      exact BlockContinues.range_done (by simp [Calls.rangeContinues]) BlockContinues.nil
  | succ count ih =>
      let nextEnv := setLocal env "i" (number start)
      have nextLocals : Locals nextEnv maximum base.length :=
        ⟨by simpa [nextEnv, setLocal] using locals.bound, by simpa [nextEnv, setLocal] using locals.array⟩
      obtain ⟨middleEnv, middleLocals, pass⟩ := divisor_pass maximum start positive nextEnv base nextLocals
        (by simp [nextEnv, setLocal])
      obtain ⟨finalEnv, finalLocals, recurse⟩ := ih (start + 1) (by omega) (by omega) middleEnv middleLocals
      refine ⟨finalEnv, finalLocals, ?_⟩
      apply BlockContinues.range_step (by simp [Calls.rangeContinues]; omega)
      apply BlockContinues.append pass
      simpa [Calls.loopContinuation, outerBody] using recurse

private theorem initialize_list (maximum : Nat) (env : Mutable.Env) (heap : Mutable.Heap)
    (bound : env "n_max" = some (number maximum)) :
    Evaluates program env (.binOp (.list [.constant (.int 0)]) .mult
      (.binOp (.name "n_max") .add (.constant (.int 1)))) heap (.ref (heap.length + 1))
      (arrayHeap (heap ++ [.list [number 0]]) (maximum + 1) (sumsBefore 1)) := by
  let base := heap ++ [.list [number 0]]
  apply Evaluates.binary (av := .ref heap.length) (bv := number (maximum + 1))
    (middle := base) (afterRight := base)
  · exact Evaluates.list (.cons Evaluates.constant .nil)
  · apply Evaluates.binary (Evaluates.lookup bound) Evaluates.constant
    simp [Mutable.binary, scalarBinary, number]
  · simp [Mutable.binary, show (BinOp.mult == BinOp.mult) = true from rfl,
      Mutable.array, Mutable.allocate, number, base, arrayHeap, table, Except.map]

/-- The sieve executes its original nested loops and returns every proper-divisor sum. -/
theorem result (maximum : Nat) (heap : Mutable.Heap) :
    ReturnsValue program "proper_divisor_sums_sieve" [number maximum] heap (.ref (heap.length + 1))
      (heap ++ [.list [number 0], .list (table (maximum + 1) ProjectEulerStatements.P21.d)]) := by
  let base := heap ++ [.list [number 0]]
  let entry := functionEnv definition.1 definition.2 [number maximum] []
  let initial := setLocal entry "s" (.ref base.length)
  have initialLocals : Locals initial maximum base.length := ⟨rfl, by simp [initial, setLocal]⟩
  obtain ⟨finalEnv, finalLocals, scanned⟩ := outer_loop (maximum / 2) 1 maximum (by omega) (by omega)
    initial base initialLocals
  have completed : table (maximum + 1) (sumsBefore (maximum / 2 + 1)) =
      table (maximum + 1) ProjectEulerStatements.P21.d :=
    table_congr (maximum + 1) _ _ (fun n bound => sumsBefore_complete _ n (by omega))
  have executed : ReturnsValue program "proper_divisor_sums_sieve" [number maximum] heap (.ref base.length)
      (arrayHeap base (maximum + 1) (sumsBefore (maximum / 2 + 1))) := by
    apply BlockReturns.function (args := definition.1) (body := definition.2) rfl rfl
    apply BlockReturns.assign (value := .ref base.length)
    · simpa only [base, List.length_append, List.length_cons, List.length_nil, Nat.add_zero] using
        initialize_list maximum entry heap rfl
    · apply BlockContinues.returns (body := [.forIn (.name "i") (.call (.name "range")
        [.constant (.int 1), .binOp (.binOp (.name "n_max") .floorDiv (.constant (.int 2))) .add (.constant (.int 1))]) outerBody []])
        (middleEnv := finalEnv) (middleHeap := arrayHeap base (maximum + 1) (sumsBefore (maximum / 2 + 1)))
      · apply BlockContinues.for_range (values := [number 1, number (maximum / 2 + 1)])
          (middle := arrayHeap base (maximum + 1) (sumsBefore 1))
        · apply Evaluates.Arguments.cons (middle := arrayHeap base (maximum + 1) (sumsBefore 1)) Evaluates.constant
          apply Evaluates.Arguments.cons (middle := arrayHeap base (maximum + 1) (sumsBefore 1))
          · apply Evaluates.binary (av := number (maximum / 2)) (bv := number 1)
              (middle := arrayHeap base (maximum + 1) (sumsBefore 1))
              (afterRight := arrayHeap base (maximum + 1) (sumsBefore 1))
            · apply Evaluates.binary (Evaluates.lookup initialLocals.bound) Evaluates.constant
              simp [Mutable.binary, scalarBinary, number]
            · exact Evaluates.constant
            · simp [Mutable.binary, scalarBinary, number]
          · exact .nil
        · rfl
        · simpa only [number, Nat.cast_add, Nat.cast_one, Int.natCast_ediv, Nat.cast_ofNat] using scanned
      · exact BlockReturns.return_value (Evaluates.lookup finalLocals.array)
  simpa only [arrayHeap, completed, base, List.append_assoc, List.cons_append, List.nil_append,
    List.length_append, List.length_cons, List.length_nil, Nat.add_zero] using executed

/-- For nonpositive maxima the sieve allocates its zero list and skips both loops. -/
theorem small_result (maximum : Int) (small : maximum ≤ 0) (heap : Mutable.Heap) :
    ReturnsValue program "proper_divisor_sums_sieve" [.scalar (.int maximum)] heap (.ref (heap.length + 1))
      (heap ++ [.list [number 0], .list (List.replicate (maximum + 1).toNat (number 0))]) := by
  let entry := functionEnv definition.1 definition.2 [.scalar (.int maximum)] []
  let base := heap ++ [.list [number 0]]
  let allocated := base ++ [.list (List.replicate (maximum + 1).toNat (number 0))]
  have allocation : Evaluates program entry (.binOp (.list [.constant (.int 0)]) .mult
      (.binOp (.name "n_max") .add (.constant (.int 1)))) heap (.ref (heap.length + 1)) allocated := by
    apply Evaluates.binary (av := .ref heap.length) (bv := .scalar (.int (maximum + 1)))
      (middle := base) (afterRight := base)
    · exact Evaluates.list (.cons Evaluates.constant .nil)
    · apply Evaluates.binary (Evaluates.lookup rfl) Evaluates.constant
      simp [Mutable.binary, scalarBinary]
    · simp [Mutable.binary, show (BinOp.mult == BinOp.mult) = true from rfl,
        Mutable.array, Mutable.allocate, number, base, allocated, Except.map]
  have executed : ReturnsValue program "proper_divisor_sums_sieve" [.scalar (.int maximum)] heap
      (.ref (heap.length + 1)) allocated := by
    apply BlockReturns.function (args := definition.1) (body := definition.2) rfl rfl
    apply BlockReturns.assign allocation
    apply BlockReturns.for_range (values := [.scalar (.int 1), .scalar (.int (maximum / 2 + 1))])
      (middle := allocated)
    · apply Evaluates.Arguments.cons (middle := allocated) Evaluates.constant
      apply Evaluates.Arguments.cons (middle := allocated)
      · apply Evaluates.binary (av := .scalar (.int (maximum / 2))) (bv := number 1)
          (middle := allocated) (afterRight := allocated)
        · apply Evaluates.binary (Evaluates.lookup rfl) Evaluates.constant
          simp [Mutable.binary, scalarBinary]
        · exact Evaluates.constant
        · simp [Mutable.binary, scalarBinary]
      · exact .nil
    · rfl
    · apply BlockReturns.range_done (by simp [Calls.rangeContinues]; omega)
      exact BlockReturns.return_value (Evaluates.lookup rfl)
  simpa only [allocated, base, List.append_assoc, List.cons_append, List.nil_append] using executed

end Sieve

/-- Empty candidate ranges still execute the sieve allocation and return zero. -/
theorem solve.small_result (limit : Int) (small : limit ≤ 1) (heap : Mutable.Heap) :
    ReturnsValue program "solve" [.scalar (.int limit)] heap (.scalar (.int 0))
      (heap ++ [.list [.scalar (.int 0)], .list (List.replicate limit.toNat (.scalar (.int 0)))]) := by
  let allocated := heap ++ [.list [.scalar (.int 0)], .list (List.replicate limit.toNat (.scalar (.int 0)))]
  apply BlockReturns.function rfl rfl
  apply BlockReturns.assign (value := .ref (heap.length + 1)) (middle := allocated)
  · apply Evaluates.call (values := [.scalar (.int (limit - 1))]) (middle := heap)
    · exact .cons (Evaluates.binary (middle := heap) (afterRight := heap) (Evaluates.lookup rfl) Evaluates.constant
        (by simp [Mutable.binary, scalarBinary])) .nil
    · simpa only [Int.sub_add_cancel] using Sieve.small_result (limit - 1) (by omega) heap
  · apply BlockReturns.assign Evaluates.constant
    apply BlockReturns.for_range (.cons Evaluates.constant (.cons (Evaluates.lookup rfl) .nil)) rfl
    apply BlockReturns.range_done (by simp [Calls.rangeContinues]; omega)
    exact BlockReturns.return_value (Evaluates.lookup rfl)

theorem solve.small_equiv (limit : Int) (small : limit ≤ 1) :
    (HeapCalls.scalarFunction program "solve" [.int limit]).Equiv
      (Calls.function P21Ast.module "naive" [.int limit]) := by
  have optimized := ReturnsValue.scalar_finishes (values := [.int limit]) (solve.small_result limit small [])
  have reference := (data.«project-euler-statements».data.py.project_euler_statements.p21.naive.empty_result limit small).finishes
  intro outcome
  constructor
  · intro finished
    rw [finished.unique optimized]
    exact reference
  · intro finished
    rw [finished.unique reference]
    exact optimized

namespace Factors
open ProjectEulerStatements.DivisorSumFactors
open ProjectEulerStatements.FactorDivisors
private abbrev number (n : Nat) : Mutable.Value := .scalar (.int n)

private def factorExpr (p : Nat) (isNamed : Bool) : Expr :=
  if isNamed then .name "p" else .constant (.int p)

private def stripBody (p : Nat) (isNamed : Bool) : List Stmt :=
  [.augAssign (.name "n") .floorDiv (factorExpr p isNamed),
   .augAssign (.name "p_pow") .mult (factorExpr p isNamed),
   .augAssign (.name "term") .add (.name "p_pow")]

private def stripCondition (p : Nat) (isNamed : Bool) : Expr :=
  .compare (.binOp (.name "n") .mod (factorExpr p isNamed)) [(.eq, .constant (.int 0))]

private def stripLoop (p : Nat) (isNamed : Bool) : Stmt :=
  .whileDo (stripCondition p isNamed) (stripBody p isNamed) []

private structure StripLocals (env : Mutable.Env) (p n exponent : Nat) (isNamed : Bool) : Prop where
  current : env "n" = some (number n)
  power : env "p_pow" = some (number (p ^ exponent))
  term : env "term" = some (number (geometric p exponent))
  factor : isNamed = true → env "p" = some (number p)

private theorem factor_evaluates (p : Nat) (isNamed : Bool) (env : Mutable.Env) (heap : Mutable.Heap)
    (factor : isNamed = true → env "p" = some (number p)) :
    Evaluates program env (factorExpr p isNamed) heap (number p) heap := by
  cases isNamed with
  | false => exact Evaluates.constant
  | true => exact Evaluates.lookup (factor rfl)

private def stepEnv (env : Mutable.Env) (p n exponent : Nat) : Mutable.Env :=
  setLocal (setLocal (setLocal env "n" (number (n / p))) "p_pow" (number (p ^ (exponent + 1))))
    "term" (number (geometric p (exponent + 1)))

private theorem strip_one (p n exponent : Nat) (isNamed : Bool) (prime : Nat.Prime p)
    (env : Mutable.Env) (heap : Mutable.Heap) (locals : StripLocals env p n exponent isNamed) :
    BlockContinues program env (stripBody p isNamed) heap (stepEnv env p n exponent) heap := by
  let divided := setLocal env "n" (number (n / p))
  let powered := setLocal divided "p_pow" (number (p ^ (exponent + 1)))
  apply BlockContinues.append (body := [.augAssign (.name "n") .floorDiv (factorExpr p isNamed)])
    (middleEnv := divided) (middleHeap := heap)
  · apply BlockContinues.augAssign locals.current (factor_evaluates p isNamed env heap locals.factor)
    simp [inplace, Mutable.binary, scalarBinary, number, prime.ne_zero, prime.pos]
  · apply BlockContinues.append (body := [.augAssign (.name "p_pow") .mult (factorExpr p isNamed)])
      (middleEnv := powered) (middleHeap := heap)
    · apply BlockContinues.augAssign (by simpa [divided, setLocal] using locals.power)
        (factor_evaluates p isNamed divided heap (by intro active; simpa [divided, setLocal] using locals.factor active))
      simp [inplace, Mutable.binary, scalarBinary, number, pow_succ]
    · apply BlockContinues.augAssign (right := number (p ^ (exponent + 1)))
        (by simpa [powered, divided, setLocal] using locals.term)
        (Evaluates.lookup (by simp [powered, setLocal]))
      simp [inplace, Mutable.binary, scalarBinary, number, geometric_succ]

private theorem strip_test (p n exponent : Nat) (isNamed : Bool) (prime : Nat.Prime p)
    (env : Mutable.Env) (heap : Mutable.Heap) (locals : StripLocals env p n exponent isNamed) :
    Evaluates program env (stripCondition p isNamed) heap (.scalar (.bool (decide (p ∣ n)))) heap := by
  apply Evaluates.compare (av := number (n % p)) (bv := number 0) (middle := heap) (afterRight := heap)
  · apply Evaluates.binary (Evaluates.lookup locals.current) (factor_evaluates p isNamed env heap locals.factor)
    simp [Mutable.binary, scalarBinary, number, prime.ne_zero, prime.pos]
  · exact Evaluates.constant
  · simp [HeapCalls.compare, scalarCompare, evalCmpOp, Int.natCast_dvd_natCast]

/-- Repeated prime division computes the entire geometric term and leaves other locals unchanged. -/
private theorem strip_loop_result (n p exponent : Nat) (isNamed : Bool) (positive : 0 < n)
    (prime : Nat.Prime p) (env : Mutable.Env) (heap : Mutable.Heap)
    (locals : StripLocals env p n exponent isNamed) :
    ∃ finalEnv, StripLocals finalEnv p (strip n p).1 (exponent + (strip n p).2) isNamed ∧
      (∀ name, name ≠ "n" → name ≠ "p_pow" → name ≠ "term" → finalEnv name = env name) ∧
      BlockContinues program env [stripLoop p isNamed] heap finalEnv heap := by
  have test := strip_test p n exponent isNamed prime env heap locals
  by_cases divides : p ∣ n
  · have quotientPositive : 0 < n / p := Nat.div_pos (Nat.le_of_dvd positive divides) prime.pos
    have nextLocals : StripLocals (stepEnv env p n exponent) p (n / p) (exponent + 1) isNamed := by
      refine ⟨by simp [stepEnv, setLocal], by simp [stepEnv, setLocal], by simp [stepEnv, setLocal], ?_⟩
      intro active
      simpa [stepEnv, setLocal] using locals.factor active
    obtain ⟨finalEnv, finalLocals, preserved, recurse⟩ := strip_loop_result (n / p) p (exponent + 1)
      isNamed quotientPositive prime (stepEnv env p n exponent) heap nextLocals
    have stripped : strip n p = ((strip (n / p) p).1, (strip (n / p) p).2 + 1) := by
      rw [strip, dif_pos ⟨prime.one_lt, positive, divides⟩]
    refine ⟨finalEnv, ?_, ?_, ?_⟩
    · simpa [stripped, Nat.add_assoc, Nat.add_comm, Nat.add_left_comm] using finalLocals
    · intro name notN notPower notTerm
      exact (preserved name notN notPower notTerm).trans (by simp [stepEnv, setLocal, notN, notPower, notTerm])
    · apply BlockContinues.while_loop test (flag := true) (by simp [Mutable.truth, scalarTruth, divides])
      apply BlockContinues.append (strip_one p n exponent isNamed prime env heap locals)
      simpa [Calls.loopContinuation, stripBody, stripLoop] using recurse
  · have stripped : strip n p = (n, 0) := by
      rw [strip, dif_neg (by tauto)]
    refine ⟨env, ?_, by intros; rfl, ?_⟩
    · simpa [stripped] using locals
    · exact BlockContinues.while_loop test (flag := false) (by simp [Mutable.truth, scalarTruth, divides]) BlockContinues.nil
termination_by n
decreasing_by exact Nat.div_lt_self positive prime.one_lt

private structure PassLocals (env : Mutable.Env) (n acc p : Nat) (isNamed : Bool) : Prop where
  current : env "n" = some (number n)
  accumulated : env "sigma" = some (number acc)
  factor : isNamed = true → env "p" = some (number p)

private def passBody (p : Nat) (isNamed : Bool) : List Stmt :=
  [.assign [.name "term"] (.constant (.int 1)),
   .assign [.name "p_pow"] (.constant (.int 1)),
   stripLoop p isNamed,
   .augAssign (.name "sigma") .mult (.name "term")]

private def passStatement (p : Nat) (isNamed : Bool) : Stmt :=
  .ifThen (stripCondition p isNamed) (passBody p isNamed) []

private theorem divisibility_test (p n acc : Nat) (isNamed : Bool) (positive : 0 < p)
    (env : Mutable.Env) (heap : Mutable.Heap) (locals : PassLocals env n acc p isNamed) :
    Evaluates program env (stripCondition p isNamed) heap (.scalar (.bool (decide (p ∣ n)))) heap := by
  apply Evaluates.compare (av := number (n % p)) (bv := number 0) (middle := heap) (afterRight := heap)
  · apply Evaluates.binary (Evaluates.lookup locals.current) (factor_evaluates p isNamed env heap locals.factor)
    simp [Mutable.binary, scalarBinary, number, positive.ne', positive]
  · exact Evaluates.constant
  · simp [HeapCalls.compare, scalarCompare, evalCmpOp, Int.natCast_dvd_natCast]

private theorem pass_body_result (p n acc : Nat) (isNamed : Bool) (positive : 0 < n)
    (prime : Nat.Prime p) (env : Mutable.Env) (heap : Mutable.Heap) (locals : PassLocals env n acc p isNamed) :
    ∃ finalEnv, PassLocals finalEnv (strip n p).1 (acc * geometric p (strip n p).2) p isNamed ∧
      (∀ name, name ≠ "n" → name ≠ "p_pow" → name ≠ "term" → name ≠ "sigma" → finalEnv name = env name) ∧
      BlockContinues program env (passBody p isNamed) heap finalEnv heap := by
  let initial := setLocal (setLocal env "term" (number 1)) "p_pow" (number 1)
  have initialLocals : StripLocals initial p n 0 isNamed := by
    refine ⟨by simpa [initial, setLocal] using locals.current, by simp [initial, setLocal, number],
      by simp [initial, setLocal], ?_⟩
    intro active
    simpa [initial, setLocal] using locals.factor active
  obtain ⟨afterStrip, stripLocals, preserved, stripped⟩ :=
    strip_loop_result n p 0 isNamed positive prime initial heap initialLocals
  have oldSigma : afterStrip "sigma" = some (number acc) := by
    rw [preserved "sigma" (by decide) (by decide) (by decide)]
    simpa [initial, setLocal] using locals.accumulated
  let finalEnv := setLocal afterStrip "sigma" (number (acc * geometric p (strip n p).2))
  refine ⟨finalEnv, ?_, ?_, ?_⟩
  · refine ⟨by simpa [finalEnv, setLocal] using stripLocals.current, by simp [finalEnv, setLocal], ?_⟩
    intro active
    simpa [finalEnv, setLocal] using stripLocals.factor active
  · intro name notN notPower notTerm notSigma
    simp only [finalEnv, setLocal, if_neg notSigma]
    rw [preserved name notN notPower notTerm]
    simp [initial, setLocal, notPower, notTerm]
  · apply BlockContinues.append (body := [.assign [.name "term"] (.constant (.int 1)),
      .assign [.name "p_pow"] (.constant (.int 1))]) (middleEnv := initial) (middleHeap := heap)
    · apply BlockContinues.append (body := [.assign [.name "term"] (.constant (.int 1))])
        (middleEnv := setLocal env "term" (number 1)) (middleHeap := heap)
      · exact BlockContinues.assign Evaluates.constant
      · exact BlockContinues.assign Evaluates.constant
    · apply BlockContinues.append stripped
      apply BlockContinues.augAssign oldSigma
        (Evaluates.lookup (by simpa only [Nat.zero_add] using stripLocals.term))
      simp [inplace, Mutable.binary, scalarBinary, number]

private structure PassEffect (n p remainder product : Nat) : Prop where
  positive : 0 < remainder
  divides : remainder ∣ n
  removed : ¬ p ∣ remainder
  summed : sigma n = product * sigma remainder

/-- A complete source trial pass either skips the candidate or removes its full prime power. -/
private theorem factor_pass (n acc p : Nat) (isNamed : Bool) (positive : 0 < n) (large : 1 < p)
    (primeWhen : p ∣ n → Nat.Prime p) (env : Mutable.Env) (heap : Mutable.Heap)
    (locals : PassLocals env n acc p isNamed) :
    ∃ remainder product finalEnv, PassEffect n p remainder product ∧
      PassLocals finalEnv remainder (acc * product) p isNamed ∧
      (∀ name, name ≠ "n" → name ≠ "p_pow" → name ≠ "term" → name ≠ "sigma" → finalEnv name = env name) ∧
      BlockContinues program env [passStatement p isNamed] heap finalEnv heap := by
  have test := divisibility_test p n acc isNamed (by omega) env heap locals
  by_cases divides : p ∣ n
  · have prime := primeWhen divides
    have spec := strip_spec n p positive prime
    obtain ⟨finalEnv, finalLocals, preserved, body⟩ := pass_body_result p n acc isNamed positive prime env heap locals
    refine ⟨(strip n p).1, geometric p (strip n p).2, finalEnv,
      ⟨spec.1, strip_dvd n p positive prime, spec.2.1, strip_sigma n p positive prime⟩,
      finalLocals, preserved, ?_⟩
    exact BlockContinues.branch test (flag := true) (by simp [Mutable.truth, scalarTruth, divides]) body
  · refine ⟨n, 1, env, ⟨positive, dvd_refl n, divides, by simp⟩, ?_, by intros; rfl, ?_⟩
    · simpa only [Nat.mul_one] using locals
    · exact BlockContinues.branch test (flag := false) (by simp [Mutable.truth, scalarTruth, divides]) BlockContinues.nil

private def oddCondition : Expr :=
  .compare (.binOp (.name "p") .mult (.name "p")) [(.ltE, .name "n")]

private def oddBody : List Stmt :=
  [passStatement 0 true, .augAssign (.name "p") .add (.constant (.int 2))]

private def oddLoop : Stmt := .whileDo oddCondition oddBody []

private structure LoopLocals (env : Mutable.Env) (original n acc p : Nat) : Prop where
  pass : PassLocals env n acc p true
  original : env "original" = some (number original)

private theorem odd_test (original n acc p : Nat) (env : Mutable.Env) (heap : Mutable.Heap)
    (locals : LoopLocals env original n acc p) :
    Evaluates program env oddCondition heap (.scalar (.bool (decide (p * p ≤ n)))) heap := by
  apply Evaluates.compare (av := number (p * p)) (bv := number n) (middle := heap) (afterRight := heap)
  · apply Evaluates.binary (Evaluates.lookup (locals.pass.factor rfl)) (Evaluates.lookup (locals.pass.factor rfl))
    simp [Mutable.binary, scalarBinary, number]
  · exact Evaluates.lookup locals.pass.current
  · simp [HeapCalls.compare, scalarCompare, evalCmpOp]
    norm_cast

/-- Trial division preserves sigma and advances beyond every possible small prime factor. -/
private theorem odd_loop_result (original n acc p : Nat) (positive : 0 < n) (large : 3 ≤ p)
    (odd : p % 2 = 1) (free : NoSmallPrime n p) (invariant : sigma original = acc * sigma n)
    (env : Mutable.Env) (heap : Mutable.Heap) (locals : LoopLocals env original n acc p) :
    ∃ remainder finalAcc next finalEnv,
      0 < remainder ∧ NoSmallPrime remainder next ∧ remainder < next * next ∧
      sigma original = finalAcc * sigma remainder ∧
      LoopLocals finalEnv original remainder finalAcc next ∧
      BlockContinues program env [oddLoop] heap finalEnv heap := by
  have test := odd_test original n acc p env heap locals
  by_cases active : p * p ≤ n
  · obtain ⟨remainder, product, afterPass, effect, passLocals, preserved, passed⟩ :=
      factor_pass n acc p true positive (by omega) (trial_prime n p (by omega) free) env heap locals.pass
    have notIncreased : remainder ≤ n := Nat.le_of_dvd positive effect.divides
    have remainingFree : NoSmallPrime remainder p := by
      intro q prime before divides
      exact free q prime before (divides.trans effect.divides)
    have nextFree := advance_free remainder p large odd remainingFree effect.removed
    have nextInvariant : sigma original = (acc * product) * sigma remainder := by
      rw [invariant, effect.summed, Nat.mul_assoc]
    let nextEnv := setLocal afterPass "p" (number (p + 2))
    have nextLocals : LoopLocals nextEnv original remainder (acc * product) (p + 2) := by
      refine ⟨⟨by simpa [nextEnv, setLocal] using passLocals.current,
        by simpa [nextEnv, setLocal] using passLocals.accumulated, by intros; simp [nextEnv, setLocal]⟩, ?_⟩
      simpa [nextEnv, setLocal, preserved "original" (by decide) (by decide) (by decide) (by decide)] using locals.original
    obtain ⟨finalN, finalAcc, finalP, finalEnv, finalPositive, finalFree, finished, finalInvariant, finalLocals, recurse⟩ :=
      odd_loop_result original remainder (acc * product) (p + 2) effect.positive (by omega)
        (by omega) nextFree nextInvariant nextEnv heap nextLocals
    refine ⟨finalN, finalAcc, finalP, finalEnv, finalPositive, finalFree, finished, finalInvariant, finalLocals, ?_⟩
    apply BlockContinues.while_loop test (flag := true) (by simp [Mutable.truth, scalarTruth, active])
    apply BlockContinues.append (body := oddBody) (middleEnv := nextEnv) (middleHeap := heap)
    · apply BlockContinues.append (body := [passStatement 0 true]) (middleEnv := afterPass) (middleHeap := heap)
      · simpa only [passStatement, passBody, stripCondition, stripLoop, stripBody, factorExpr] using passed
      · apply BlockContinues.augAssign (passLocals.factor rfl) Evaluates.constant
        simp [inplace, Mutable.binary, scalarBinary, number]
    · simpa [Calls.loopContinuation, oddBody, oddLoop, passStatement, passBody, stripLoop, stripBody] using recurse
  · refine ⟨n, acc, p, env, positive, free, by omega, invariant, locals, ?_⟩
    exact BlockContinues.while_loop test (flag := false) (by simp [Mutable.truth, scalarTruth, active]) BlockContinues.nil
termination_by n + 1 - p
decreasing_by
  have bound : p ≤ n := (Nat.le_mul_self p).trans active
  omega

private def tail : List Stmt :=
  [.ifThen (.compare (.name "n") [(.gt, .constant (.int 1))])
    [.augAssign (.name "sigma") .mult (.binOp (.constant (.int 1)) .add (.name "n"))] [],
   .return (.binOp (.name "sigma") .sub (.name "original"))]

private theorem return_result (original : Nat) (positive : 0 < original) (env : Mutable.Env)
    (heap : Mutable.Heap) (accumulated : env "sigma" = some (number (sigma original)))
    (originalValue : env "original" = some (number original)) :
    BlockReturns program env [.return (.binOp (.name "sigma") .sub (.name "original"))]
      heap (number (ProjectEulerStatements.P21.d original)) heap := by
  apply BlockReturns.return_value
  apply Evaluates.binary (Evaluates.lookup accumulated) (Evaluates.lookup originalValue)
  simp [Mutable.binary, scalarBinary, number, sigma_eq_self_add original positive]

private theorem finish_result (original n acc p : Nat) (positive : 0 < original) (remaining : 0 < n)
    (free : NoSmallPrime n p) (finished : n < p * p) (invariant : sigma original = acc * sigma n)
    (env : Mutable.Env) (heap : Mutable.Heap) (locals : LoopLocals env original n acc p) :
    BlockReturns program env tail heap (number (ProjectEulerStatements.P21.d original)) heap := by
  have test : Evaluates program env (.compare (.name "n") [(.gt, .constant (.int 1))]) heap
      (.scalar (.bool (decide (1 < n)))) heap := by
    apply Evaluates.compare (Evaluates.lookup locals.pass.current) Evaluates.constant
    simp [HeapCalls.compare, scalarCompare, evalCmpOp]
  have remainingSigma := finish_sigma n p remaining free finished
  by_cases large : 1 < n
  · have fullSigma : sigma original = acc * (1 + n) := by simpa [large, remainingSigma] using invariant
    apply BlockReturns.branch test (flag := true) (by simp [Mutable.truth, scalarTruth, large])
    apply BlockReturns.augAssign (value := number (sigma original)) (right := number (1 + n)) (middle := heap) (afterOperation := heap) locals.pass.accumulated
    · apply Evaluates.binary Evaluates.constant (Evaluates.lookup locals.pass.current)
      simp [Mutable.binary, scalarBinary, number]
    · simp [inplace, Mutable.binary, scalarBinary, number, fullSigma]
    · exact return_result original positive _ heap (by simp [setLocal]) (by simpa [setLocal] using locals.original)
  · have fullSigma : sigma original = acc := by simpa [large, remainingSigma] using invariant
    apply BlockReturns.branch test (flag := false) (by simp [Mutable.truth, scalarTruth, large])
    exact return_result original positive env heap (by simpa only [fullSigma] using locals.pass.accumulated) locals.original

private def definition : List Arg × List Stmt :=
  (findFunctionIn program.body "proper_divisor_sum_factorization").getD ([], [])

/-- The source factorization computes the mathematical divisor sum without changing the heap. -/
theorem result (original : Nat) (heap : Mutable.Heap) :
    ReturnsValue program "proper_divisor_sum_factorization" [number original] heap
      (number (ProjectEulerStatements.P21.d original)) heap := by
  by_cases small : original ≤ 1
  · have empty : ProjectEulerStatements.P21.d original = 0 := by
      interval_cases original <;> simp [ProjectEulerStatements.P21.d, ProjectEulerStatements.P21.properDivisors]
    simpa only [empty] using factorization.small_result original (by omega) heap
  · have positive : 0 < original := by omega
    let initial := setLocal (setLocal (functionEnv definition.1 definition.2 [number original] [])
      "original" (number original)) "sigma" (number 1)
    have initialLocals : PassLocals initial original 1 2 false := ⟨rfl, rfl, by simp⟩
    obtain ⟨remainder, product, afterPass, effect, passLocals, preserved, passed⟩ :=
      factor_pass original 1 2 false positive (by decide) (fun _ => Nat.prime_two) initial heap initialLocals
    let beforeLoop := setLocal afterPass "p" (number 3)
    have loopLocals : LoopLocals beforeLoop original remainder product 3 := by
      refine ⟨⟨by simpa [beforeLoop, setLocal] using passLocals.current,
        by simpa [beforeLoop, setLocal] using passLocals.accumulated, by intros; simp [beforeLoop, setLocal]⟩, ?_⟩
      simp only [beforeLoop, setLocal, show ("original" : String) ≠ "p" by decide, if_false]
      rw [preserved "original" (by decide) (by decide) (by decide) (by decide)]
      rfl
    obtain ⟨finalN, finalAcc, finalP, finalEnv, finalPositive, finalFree, finished, finalInvariant, finalLocals, loop⟩ :=
      odd_loop_result original remainder product 3 effect.positive (by decide) (by decide)
        (initial_free remainder effect.removed) effect.summed beforeLoop heap loopLocals
    apply BlockReturns.function (args := definition.1) (body := definition.2) rfl rfl
    apply BlockReturns.branch (value := .scalar (.bool false)) (flag := false) (middle := heap)
    · apply Evaluates.compare (Evaluates.lookup rfl) Evaluates.constant
      simp [HeapCalls.compare, scalarCompare, evalCmpOp, small]
    · rfl
    · apply BlockReturns.assign (Evaluates.lookup rfl)
      apply BlockReturns.assign Evaluates.constant
      apply BlockContinues.returns (body := [passStatement 2 false]) passed
      apply BlockReturns.assign Evaluates.constant
      apply BlockContinues.returns (body := [oddLoop]) loop
      exact finish_result original finalN finalAcc finalP positive finalPositive finalFree finished
        finalInvariant finalEnv heap finalLocals

end Factors
namespace Cached
open ProjectEulerStatements.P21
private abbrev number (n : Nat) : Mutable.Value := .scalar (.int n)

private theorem test (x size address : Nat) (env : Mutable.Env) (heap : Mutable.Heap)
    (input : env "x" = some (number x)) (array : env "divsum" = some (.ref address))
    (found : heap[address]? = some (.list (Sieve.table size d))) :
    Evaluates program env (.compare (.constant (.int 0))
      [(.ltE, .name "x"), (.lt, .builtinCall .len [.name "divsum"])])
      heap (.scalar (.bool (decide (x < size)))) heap := by
  apply Evaluates.chain Evaluates.constant
  apply Evaluates.Compares.next (compared := .scalar (.bool true)) (afterComparison := heap)
    (Evaluates.lookup input)
  · simp [HeapCalls.compare, scalarCompare, evalCmpOp]
  · rfl
  · apply Evaluates.Compares.last (value := number size) (middle := heap)
    · apply Evaluates.builtin (.cons (Evaluates.lookup array) .nil)
      simp [primitive, Mutable.length, found]
    · simp [HeapCalls.compare, scalarCompare, evalCmpOp]

private def definition : List Arg × List Stmt :=
  (findFunctionIn program.body "__formalsw_local_solve_d").getD ([], [])

/-- The local helper reads the captured sieve or computes the same value by factorization. -/
theorem result (x size address : Nat) (heap : Mutable.Heap)
    (found : heap[address]? = some (.list (Sieve.table size d))) :
    ReturnsValue program "__formalsw_local_solve_d" [number x, .ref address]
      heap (number (d x)) heap := by
  apply BlockReturns.function rfl rfl
  have condition := test x size address (functionEnv definition.1 definition.2 [number x, .ref address] []) heap rfl rfl found
  by_cases inside : x < size
  · apply BlockReturns.branch condition (flag := true) (by simp [Mutable.truth, scalarTruth, inside])
    apply BlockReturns.return_value
    apply Evaluates.subscript (by intro a b c; simp) (Evaluates.lookup rfl) (Evaluates.lookup rfl)
    have item := Sieve.table_lookup size d x inside
    simpa only [(List.getElem?_eq_some_iff.mp item).2] using
      Mutable.getItem_list heap address x _ found (by simpa using inside)
  · apply BlockReturns.branch condition (flag := false) (by simp [Mutable.truth, scalarTruth, inside])
    apply BlockReturns.return_value
    exact Evaluates.call (.cons (Evaluates.lookup rfl) .nil) (Factors.result x heap)

end Cached

namespace Sum
open ProjectEulerStatements.P21
private abbrev number (n : Nat) : Mutable.Value := .scalar (.int n)

private def condition : Expr := .boolOp .and
  [.compare (.name "b") [(.notEq, .name "a")],
   .compare (.call (.name "__formalsw_local_solve_d") [.name "b", .name "divsum"])
     [(.eq, .name "a")]]

private def body : List Stmt :=
  [.assign [.name "b"] (.call (.name "__formalsw_local_solve_d") [.name "a", .name "divsum"]),
   .ifThen condition [.augAssign (.name "total") .add (.name "a")] []]

private def term (n : Nat) : Nat := if isAmicable n then n else 0
private def selected (start count : Nat) : Nat := ((List.range' start count).filter (fun n => decide (isAmicable n))).sum

private structure Locals (env : Mutable.Env) (limit address total : Nat) : Prop where
  bound : env "limit" = some (number limit)
  array : env "divsum" = some (.ref address)
  accumulated : env "total" = some (number total)

private theorem condition_result (a size address : Nat) (env : Mutable.Env) (heap : Mutable.Heap)
    (input : env "a" = some (number a)) (partner : env "b" = some (number (d a)))
    (array : env "divsum" = some (.ref address)) (found : heap[address]? = some (.list (Sieve.table size d))) :
    Evaluates program env condition heap (.scalar (.bool (decide (isAmicable a)))) heap := by
  have first : Evaluates program env (.compare (.name "b") [(.notEq, .name "a")]) heap
      (.scalar (.bool (decide (d a ≠ a)))) heap := by
    apply Evaluates.compare (Evaluates.lookup partner) (Evaluates.lookup input)
    simp [HeapCalls.compare, scalarCompare, evalCmpOp]
  by_cases different : d a ≠ a
  · apply Evaluates.boolean_next first (flag := true) (by simp [Mutable.truth, scalarTruth, different]) rfl
    apply Evaluates.boolean_last (flag := decide (isAmicable a))
    · apply Evaluates.compare (av := number (d (d a))) (bv := number a) (middle := heap) (afterRight := heap)
      · exact Evaluates.call (.cons (Evaluates.lookup partner) (.cons (Evaluates.lookup array) .nil))
          (Cached.result (d a) size address heap found)
      · exact Evaluates.lookup input
      · simp [HeapCalls.compare, scalarCompare, evalCmpOp, isAmicable, different]
    · rfl
  · have rejected : decide (isAmicable a) = false := by simp [isAmicable, different]
    rw [rejected]
    exact Evaluates.boolean_stop (by simpa only [decide_eq_false different] using first) rfl rfl

private theorem candidate (a limit address total : Nat) (env : Mutable.Env) (heap : Mutable.Heap)
    (locals : Locals env limit address total) (input : env "a" = some (number a))
    (found : heap[address]? = some (.list (Sieve.table limit d))) :
    ∃ finalEnv, Locals finalEnv limit address (total + term a) ∧
      BlockContinues program env body heap finalEnv heap := by
  let afterB := setLocal env "b" (number (d a))
  have computed : BlockContinues program env [body[0]] heap afterB heap :=
    BlockContinues.assign (Evaluates.call (.cons (Evaluates.lookup input) (.cons (Evaluates.lookup locals.array) .nil))
      (Cached.result a limit address heap found))
  have nextInput : afterB "a" = some (number a) := by simpa [afterB, setLocal] using input
  have nextArray : afterB "divsum" = some (.ref address) := by simpa [afterB, setLocal] using locals.array
  have test := condition_result a limit address afterB heap nextInput (by simp [afterB, setLocal]) nextArray found
  by_cases accepted : isAmicable a
  · let finalEnv := setLocal afterB "total" (number (total + a))
    refine ⟨finalEnv, ?_, ?_⟩
    · refine ⟨by simpa [finalEnv, afterB, setLocal] using locals.bound,
        by simpa [finalEnv, afterB, setLocal] using locals.array, ?_⟩
      simp [finalEnv, setLocal, term, accepted]
    · apply BlockContinues.append (body := [body[0]]) computed
      apply BlockContinues.branch test (flag := true) (by simp [Mutable.truth, scalarTruth, accepted])
      apply BlockContinues.augAssign (by simpa [afterB, setLocal] using locals.accumulated) (Evaluates.lookup nextInput)
      simp [inplace, Mutable.binary, scalarBinary, number]
  · refine ⟨afterB, ?_, ?_⟩
    · refine ⟨by simpa [afterB, setLocal] using locals.bound, nextArray, ?_⟩
      simpa [afterB, setLocal, term, accepted] using locals.accumulated
    · apply BlockContinues.append (body := [body[0]]) computed
      exact BlockContinues.branch test (flag := false) (by simp [Mutable.truth, scalarTruth, accepted]) BlockContinues.nil

private theorem loop (count start limit address total : Nat) (stop : start + count = limit)
    (env : Mutable.Env) (heap : Mutable.Heap) (locals : Locals env limit address total)
    (found : heap[address]? = some (.list (Sieve.table limit d))) :
    ∃ finalEnv, Locals finalEnv limit address (total + selected start count) ∧
      BlockContinues program env [.rangeStepLoop "a" start limit 1 body []] heap finalEnv heap := by
  induction count generalizing start total env with
  | zero =>
      refine ⟨env, by simpa [selected] using locals, ?_⟩
      exact BlockContinues.range_done (by simp [Calls.rangeContinues]; omega) BlockContinues.nil
  | succ count ih =>
      let nextEnv := setLocal env "a" (number start)
      have nextLocals : Locals nextEnv limit address total :=
        ⟨by simpa [nextEnv, setLocal] using locals.bound,
         by simpa [nextEnv, setLocal] using locals.array,
         by simpa [nextEnv, setLocal] using locals.accumulated⟩
      obtain ⟨afterPass, passLocals, passed⟩ := candidate start limit address total nextEnv heap nextLocals
        (by simp [nextEnv, setLocal]) found
      obtain ⟨finalEnv, finalLocals, recurse⟩ := ih (start + 1) (total + term start) (by omega) afterPass passLocals
      refine ⟨finalEnv, ?_, ?_⟩
      · by_cases accepted : isAmicable start
        · simpa [selected, List.range'_succ, term, accepted, Nat.add_assoc] using finalLocals
        · simpa [selected, List.range'_succ, term, accepted] using finalLocals
      · apply BlockContinues.range_step (by simp [Calls.rangeContinues]; omega)
        apply BlockContinues.append passed
        simpa [Calls.loopContinuation, body] using recurse

private theorem selected_eq (limit : Nat) (large : 2 ≤ limit) :
    selected 2 (limit - 2) = naive limit := by
  have same : ((List.range' 2 (limit - 2)).filter (fun n => decide (isAmicable n))).toFinset =
      (Finset.Icc 2 (limit - 1)).filter isAmicable := by
    ext n
    simp only [List.mem_toFinset, List.mem_filter, List.mem_range', Finset.mem_filter, Finset.mem_Icc,
      decide_eq_true_eq]
    constructor
    · rintro ⟨⟨k, lower, rfl⟩, accepted⟩
      exact ⟨⟨by omega, by omega⟩, accepted⟩
    · rintro ⟨⟨lower, upper⟩, accepted⟩
      exact ⟨⟨n - 2, by omega, by omega⟩, accepted⟩
  have interval : Finset.Icc 1 (limit - 1) = insert 1 (Finset.Icc 2 (limit - 1)) := by
    ext n
    simp only [Finset.mem_Icc, Finset.mem_insert]
    omega
  have oneNotAmicable : ¬ isAmicable 1 := by
    simp [isAmicable, d, properDivisors]
  rw [naive, interval]
  simp only [Finset.filter_insert, oneNotAmicable, if_false]
  rw [← same, List.sum_toFinset]
  · simp [selected]
  · exact (List.nodup_range' (s := 2) (n := limit - 2)).filter _

private def definition : List Arg × List Stmt :=
  (findFunctionIn program.body "solve").getD ([], [])

/-- The original solver's sieve, captured helper, and outer loop implement the finite statement. -/
theorem result (limit : Nat) (large : 2 ≤ limit) (heap : Mutable.Heap) :
    ReturnsValue program "solve" [number limit] heap (number (naive limit))
      (heap ++ [.list [number 0], .list (Sieve.table limit d)]) := by
  let allocated := heap ++ [.list [number 0], .list (Sieve.table limit d)]
  let initial := setLocal (setLocal (functionEnv definition.1 definition.2 [number limit] [])
    "divsum" (.ref (heap.length + 1))) "total" (number 0)
  have initialLocals : Locals initial limit (heap.length + 1) 0 := ⟨rfl, rfl, rfl⟩
  have found : allocated[heap.length + 1]? = some (.list (Sieve.table limit d)) := by
    simp [allocated]
  obtain ⟨finalEnv, finalLocals, scanned⟩ := loop (limit - 2) 2 limit (heap.length + 1) 0 (by omega)
    initial allocated initialLocals found
  apply BlockReturns.function (args := definition.1) (body := definition.2) rfl rfl
  apply BlockReturns.assign (value := .ref (heap.length + 1)) (middle := allocated)
  · apply Evaluates.call (values := [number (limit - 1)]) (middle := heap)
    · apply Evaluates.Arguments.cons (middle := heap)
      · apply Evaluates.binary (Evaluates.lookup rfl) Evaluates.constant
        simp [Mutable.binary, scalarBinary, number, Int.natCast_sub (by omega : 1 ≤ limit)]
      · exact .nil
    · simpa only [Nat.sub_add_cancel (by omega : 1 ≤ limit)] using Sieve.result (limit - 1) heap
  · apply BlockReturns.assign Evaluates.constant
    apply BlockContinues.returns (body := [.forIn (.name "a") (.call (.name "range")
      [.constant (.int 2), .name "limit"]) body []]) (middleEnv := finalEnv) (middleHeap := allocated)
    · exact BlockContinues.for_range (.cons Evaluates.constant (.cons (Evaluates.lookup initialLocals.bound) .nil)) rfl scanned
    · apply BlockReturns.return_value
      exact Evaluates.lookup (by simpa only [Nat.zero_add, selected_eq limit large] using finalLocals.accumulated)

end Sum

/-- Every integer input agrees with the independently proved naive Python implementation. -/
theorem solve.«Equiv(naive)» (limit : Int) :
    (HeapCalls.scalarFunction program "solve" [.int limit]).Equiv
      (Calls.function P21Ast.module "naive" [.int limit]) := by
  by_cases small : limit ≤ 1
  · exact solve.small_equiv limit small
  · lift limit to Nat using (by omega : 0 ≤ limit)
    have optimized := ReturnsValue.scalar_finishes (values := [.int limit]) (Sum.result limit (by omega) [])
    have reference := (data.«project-euler-statements».data.py.project_euler_statements.p21.naive.result limit).finishes
    intro outcome
    constructor
    · intro finished
      rw [finished.unique optimized]
      exact reference
    · intro finished
      rw [finished.unique reference]
      exact optimized

theorem solve.«Supported» :
    (∀ limit : Int, (HeapCalls.scalarFunction program "solve" [.int limit]).Supported) ∧
    (∀ limit : Int, (Calls.function P21Ast.module "naive" [.int limit]).Supported) := by
  constructor
  · intro limit
    by_cases small : limit ≤ 1
    · exact (ReturnsValue.scalar_returns (values := [.int limit])
        (solve.small_result limit small [])).supported
    · lift limit to Nat using (by omega : 0 ≤ limit)
      exact (ReturnsValue.scalar_returns (values := [.int limit]) (Sum.result limit (by omega) [])).supported
  · intro limit
    exact (data.«project-euler-statements».data.py.project_euler_statements.p21.naive.«Returns» limit).supported

end solvers.«21»
