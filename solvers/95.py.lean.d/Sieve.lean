import «.formalsw».solvers.«95.py.ast»
import lang.py.FormalSW.HeapCalls
import ProjectEulerStatements.Lib.DivisorSieve

namespace solvers.«95»
open FormalSW.PyAst HeapCalls
private abbrev program := Solvers.P95Ast.module
set_option Elab.async false

namespace Sieve
open ProjectEulerStatements.DivisorSieve

abbrev number (n : Nat) : Mutable.Value := .scalar (.int n)

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
  (findFunctionIn program.body "compute_sum_proper_divisors").getD ([], [])

private def outerBody : List Stmt :=
  match ((executableBody definition.2)[1]? : Option Stmt) with
  | some (.forIn _ _ body _) => body
  | _ => []

private def innerBody : List Stmt :=
  match (outerBody[0]? : Option Stmt) with
  | some (.forIn _ _ body _) => body
  | _ => []

private structure Locals (env : Mutable.Env) (maximum address : Nat) : Prop where
  bound : env "limit" = some (number maximum)
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
      (Nat.dvd_add multiple (dvd_refl i)) nextEnv base nextLocals nextDivisor
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
    (bound : env "limit" = some (number maximum)) :
    Evaluates program env (.binOp (.list [.constant (.int 0)]) .mult
      (.binOp (.name "limit") .add (.constant (.int 1)))) heap (.ref (heap.length + 1))
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
    ReturnsValue program "compute_sum_proper_divisors" [number maximum] heap (.ref (heap.length + 1))
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
  have executed : ReturnsValue program "compute_sum_proper_divisors" [number maximum] heap (.ref base.length)
      (arrayHeap base (maximum + 1) (sumsBefore (maximum / 2 + 1))) := by
    apply BlockReturns.function (args := definition.1) (body := definition.2) rfl rfl
    apply BlockReturns.assign (value := .ref base.length)
    · simpa only [base, List.length_append, List.length_cons, List.length_nil, Nat.add_zero] using
        initialize_list maximum entry heap rfl
    · apply BlockContinues.returns (body := [.forIn (.name "i") (.call (.name "range")
        [.constant (.int 1), .binOp (.binOp (.name "limit") .floorDiv (.constant (.int 2))) .add (.constant (.int 1))]) outerBody []])
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
    ReturnsValue program "compute_sum_proper_divisors" [.scalar (.int maximum)] heap (.ref (heap.length + 1))
      (heap ++ [.list [number 0], .list (List.replicate (maximum + 1).toNat (number 0))]) := by
  let entry := functionEnv definition.1 definition.2 [.scalar (.int maximum)] []
  let base := heap ++ [.list [number 0]]
  let allocated := base ++ [.list (List.replicate (maximum + 1).toNat (number 0))]
  have allocation : Evaluates program entry (.binOp (.list [.constant (.int 0)]) .mult
      (.binOp (.name "limit") .add (.constant (.int 1)))) heap (.ref (heap.length + 1)) allocated := by
    apply Evaluates.binary (av := .ref heap.length) (bv := .scalar (.int (maximum + 1)))
      (middle := base) (afterRight := base)
    · exact Evaluates.list (.cons Evaluates.constant .nil)
    · apply Evaluates.binary (Evaluates.lookup rfl) Evaluates.constant
      simp [Mutable.binary, scalarBinary]
    · simp [Mutable.binary, show (BinOp.mult == BinOp.mult) = true from rfl,
        Mutable.array, Mutable.allocate, number, base, allocated, Except.map]
  have executed : ReturnsValue program "compute_sum_proper_divisors" [.scalar (.int maximum)] heap
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

end solvers.«95»
