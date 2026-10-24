import «.formalsw».solvers.«23.py.ast»
import data.«project-euler-statements».data.py.project_euler_statements.«p23.py»
import FormalSW
import ProjectEulerStatements.Lib.DivisorSieve
import ProjectEulerStatements.Lib.AbundantSums

namespace solvers.«23»
open FormalSW.PyAst FormalSW.PyAst.HeapCalls
private abbrev program := Solvers.P23Ast.module

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
  (findFunctionIn program.body "sum_proper_divisors_sieve").getD ([], [])

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
  array : env "sums" = some (.ref address)

/-- The source augmented subscription performs one exact in-place table update. -/
private theorem increment (size i j address : Nat) (inside : j < size) (value : Nat → Nat)
    (env : Mutable.Env) (heap : Mutable.Heap)
    (array : env "sums" = some (.ref address)) (divisor : env "d" = some (number i))
    (index : env "m" = some (number j))
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
    (locals : Locals env maximum base.length) (divisor : env "d" = some (number i)) :
    ∃ finalEnv, Locals finalEnv maximum base.length ∧ finalEnv "d" = some (number i) ∧
      BlockContinues program env [.rangeStepLoop "m" j (maximum + 1) i innerBody []]
        (arrayHeap base (maximum + 1) (row (sumsBefore i) i j)) finalEnv
        (arrayHeap base (maximum + 1) (sumsBefore (i + 1))) := by
  by_cases active : j < maximum + 1
  · let nextEnv := setLocal env "m" (number j)
    have nextLocals : Locals nextEnv maximum base.length := by
      exact ⟨by simpa [nextEnv, setLocal] using locals.bound, by simpa [nextEnv, setLocal] using locals.array⟩
    have nextDivisor : nextEnv "d" = some (number i) := by simpa [nextEnv, setLocal] using divisor
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
    (locals : Locals env maximum base.length) (divisor : env "d" = some (number i)) :
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
      BlockContinues program env [.rangeStepLoop "d" start (maximum / 2 + 1) 1 outerBody []]
        (arrayHeap base (maximum + 1) (sumsBefore start)) finalEnv
        (arrayHeap base (maximum + 1) (sumsBefore (maximum / 2 + 1))) := by
  induction count generalizing start env with
  | zero =>
      have same : start = maximum / 2 + 1 := by omega
      refine ⟨env, locals, ?_⟩
      rw [same]
      exact BlockContinues.range_done (by simp [Calls.rangeContinues]) BlockContinues.nil
  | succ count ih =>
      let nextEnv := setLocal env "d" (number start)
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
    ReturnsValue program "sum_proper_divisors_sieve" [number maximum] heap (.ref (heap.length + 1))
      (heap ++ [.list [number 0], .list (table (maximum + 1) ProjectEulerStatements.P21.d)]) := by
  let base := heap ++ [.list [number 0]]
  let entry := functionEnv definition.1 definition.2 [number maximum] []
  let initial := setLocal entry "sums" (.ref base.length)
  have initialLocals : Locals initial maximum base.length := ⟨rfl, by simp [initial, setLocal]⟩
  obtain ⟨finalEnv, finalLocals, scanned⟩ := outer_loop (maximum / 2) 1 maximum (by omega) (by omega)
    initial base initialLocals
  have completed : table (maximum + 1) (sumsBefore (maximum / 2 + 1)) =
      table (maximum + 1) ProjectEulerStatements.P21.d :=
    table_congr (maximum + 1) _ _ (fun n bound => sumsBefore_complete _ n (by omega))
  have executed : ReturnsValue program "sum_proper_divisors_sieve" [number maximum] heap (.ref base.length)
      (arrayHeap base (maximum + 1) (sumsBefore (maximum / 2 + 1))) := by
    apply BlockReturns.function (args := definition.1) (body := definition.2) rfl rfl
    apply BlockReturns.assign (value := .ref base.length)
    · simpa only [base, List.length_append, List.length_cons, List.length_nil, Nat.add_zero] using
        initialize_list maximum entry heap rfl
    · apply BlockContinues.returns (body := [.forIn (.name "d") (.call (.name "range")
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

end Sieve

namespace AbundantList
private abbrev number (n : Nat) : Mutable.Value := .scalar (.int n)
private abbrev encode (items : List Nat) : List Mutable.Value := items.map number
private abbrev test : Expr := .compare (.subscript (.name "sums") (.name "n")) [(.gt, .name "n")]
private abbrev body (address : Nat) : List Stmt := [.ifThen test [.comprehensionAppend address (.name "n")] []]

private theorem tested (limit n address : Nat) (inside : n ≤ limit) (heap : Mutable.Heap)
    (found : heap[address]? = some (.list (Sieve.table (limit + 1) ProjectEulerStatements.P21.d)))
    (env : HeapCalls.Env) (getSums : env "sums" = some (.ref address)) (getN : env "n" = some (number n)) :
    Evaluates program env test heap (.scalar (.bool (decide (ProjectEulerStatements.P23.isAbundant n)))) heap := by
  apply Evaluates.compare (av := number (ProjectEulerStatements.P21.d n)) (bv := number n)
    (middle := heap) (afterRight := heap)
  · apply Evaluates.subscript (by intro a b c impossible; cases impossible) (Evaluates.lookup getSums) (Evaluates.lookup getN)
    have item := Sieve.table_lookup (limit + 1) ProjectEulerStatements.P21.d n (by omega)
    simpa only [(List.getElem?_eq_some_iff.mp item).2] using
      Mutable.getItem_list heap address n _ found (by simpa using (show n < limit + 1 by omega))
  · exact Evaluates.lookup getN
  · simp [HeapCalls.compare, scalarCompare, evalCmpOp, ProjectEulerStatements.P23.isAbundant]
    rfl

private theorem loop (limit count start address : Nat) (stop : start + count = limit + 1)
    (heap : Mutable.Heap) (found : heap[address]? = some (.list (Sieve.table (limit + 1) ProjectEulerStatements.P21.d)))
    (collected : List Nat) (env : HeapCalls.Env) (getSums : env "sums" = some (.ref address)) :
    ∃ finalEnv, BlockContinues program env [.rangeStepLoop "n" start (limit + 1) 1 (body heap.length) []]
      (heap ++ [.list (encode collected)]) finalEnv
      (heap ++ [.list (encode (collected ++
        (List.range' start count).filter (fun n => decide (ProjectEulerStatements.P23.isAbundant n))))]) := by
  have addressBound := (List.getElem?_eq_some_iff.mp found).1
  induction count generalizing start collected env with
  | zero =>
      refine ⟨env, ?_⟩
      have same : start = limit + 1 := by omega
      subst start
      simpa using BlockContinues.range_done (module := program) (body := body heap.length)
        (by simp [Calls.rangeContinues]) BlockContinues.nil
  | succ count ih =>
      let boundEnv := setLocal env "n" (number start)
      have savedSums : boundEnv "sums" = some (.ref address) := by simpa [boundEnv, setLocal] using getSums
      have evaluated := tested limit start address (by omega) (heap ++ [.list (encode collected)])
        (by simpa only [List.getElem?_append_left addressBound] using found) boundEnv savedSums rfl
      let nextItems := if ProjectEulerStatements.P23.isAbundant start then collected ++ [start] else collected
      have executed : BlockContinues program boundEnv (body heap.length)
          (heap ++ [.list (encode collected)]) boundEnv (heap ++ [.list (encode nextItems)]) := by
        by_cases abundant : ProjectEulerStatements.P23.isAbundant start
        · apply BlockContinues.branch (flag := true) (by simpa [abundant] using evaluated) rfl
          have emitted := BlockContinues.comprehension_append (globals := []) (module := program)
            (env := boundEnv) (address := heap.length) (heap := heap ++ [.list (encode collected)])
            (Evaluates.lookup (name := "n") rfl)
            (by simp : (heap ++ [.list (encode collected)])[heap.length]? = some (.list (encode collected)))
          simpa [nextItems, abundant, encode] using emitted
        · simpa [body, nextItems, abundant] using
            BlockContinues.branch (yes := [.comprehensionAppend heap.length (.name "n")]) (no := [])
              (flag := false) (by simpa [abundant] using evaluated) rfl BlockContinues.nil
      obtain ⟨finalEnv, rest⟩ := ih (start + 1) (by omega) nextItems boundEnv savedSums
      refine ⟨finalEnv, ?_⟩
      apply BlockContinues.range_step (by simp [Calls.rangeContinues]; omega)
      by_cases abundant : ProjectEulerStatements.P23.isAbundant start
      all_goals simpa [body, Calls.loopContinuation, Calls.loopControl, Calls.loopControlStmt,
        nextItems, abundant, List.range'_succ, List.append_assoc] using executed.append rest

def expression : Expr := .listComp (.name "n")
  [("n", .call (.name "range") [.constant (.int 12), .binOp (.name "limit") .add (.constant (.int 1))], [test])]

/-- The optimized comprehension reads the proven sieve and produces all abundant numbers in order. -/
theorem result (limit address : Nat) (large : 12 ≤ limit) (heap : Mutable.Heap)
    (found : heap[address]? = some (.list (Sieve.table (limit + 1) ProjectEulerStatements.P21.d)))
    (env : HeapCalls.Env) (getLimit : env "limit" = some (number limit))
    (getSums : env "sums" = some (.ref address)) :
    Evaluates program env expression heap (.ref heap.length)
      (heap ++ [.list (encode (ProjectEulerStatements.AbundantSums.values limit))]) := by
  obtain ⟨finalEnv, executed⟩ := loop limit (limit - 11) 12 address (by omega) heap found []
    (comprehensionEnv env [("n", .call (.name "range")
      [.constant (.int 12), .binOp (.name "limit") .add (.constant (.int 1))], [test])])
    (by simpa [comprehensionEnv, setLocal] using getSums)
  apply Evaluates.list_comp_range (finalEnv := finalEnv)
    (.cons Evaluates.constant (.cons (Evaluates.binary (Evaluates.lookup getLimit) Evaluates.constant rfl) .nil)) rfl
  simpa [ProjectEulerStatements.AbundantSums.values] using executed

end AbundantList

namespace Marking
open ProjectEulerStatements.AbundantSums
private abbrev number (n : Nat) : Mutable.Value := .scalar (.int n)
private abbrev encode (items : List Nat) : List Mutable.Value := items.map number

def flags (size : Nat) (marked : Finset Nat) : List UInt8 :=
  (List.range size).map (fun n => if n ∈ marked then 1 else 0)

@[simp] theorem flags_length (size : Nat) (marked : Finset Nat) : (flags size marked).length = size := by
  simp [flags]

theorem flags_get (size : Nat) (marked : Finset Nat) (n : Nat) (inside : n < size) :
    (flags size marked)[n]? = some (if n ∈ marked then 1 else 0) := by
  simp [flags, inside]

theorem flags_set (size : Nat) (marked : Finset Nat) (n : Nat) (inside : n < size) :
    (flags size marked).set n 1 = flags size (insert n marked) := by
  apply List.ext_getElem?
  intro k
  by_cases bounded : k < size
  · by_cases same : n = k
    · subst k
      rw [List.getElem?_set_self (by simpa using inside), flags_get size _ n inside]
      simp
    · rw [List.getElem?_set_ne same, flags_get size marked k bounded, flags_get size _ k bounded]
      simp [Ne.symm same]
  · have outside : size ≤ k := by omega
    simp only [List.getElem?_eq_none (by simpa using outside : ((flags size marked).set n 1).length ≤ k),
      List.getElem?_eq_none (by simpa using outside : (flags size (insert n marked)).length ≤ k)]

theorem flags_congr (size : Nat) (left right : Finset Nat)
    (same : ∀ n, n < size → (n ∈ left ↔ n ∈ right)) : flags size left = flags size right := by
  apply List.map_congr_left
  intro n member
  simp only [same n (List.mem_range.mp member)]

private abbrev workHeap (base : Mutable.Heap) (limit : Nat) (marked : Finset Nat) : Mutable.Heap :=
  base ++ [.bytearray (flags (limit + 1) marked)]

private structure Locals (env : HeapCalls.Env) (limit length address buffer : Nat) : Prop where
  bound : env "lim" = some (number limit)
  length : env "L" = some (number length)
  values : env "abund" = some (.ref address)
  buffer : env "can" = some (.ref buffer)

private def innerBody : List Stmt :=
  [.assign [.name "s"] (.binOp (.name "a") .add (.subscript (.name "abund") (.name "j"))),
   .ifThen (.compare (.name "s") [(.gt, .name "lim")]) [.breakLoop] [],
   .assign [.subscript (.name "can") (.name "s")] (.constant (.int 1))]

private theorem read_item (items : List Nat) (position value address : Nat) (indexName : String)
    (heap : Mutable.Heap) (found : heap[address]? = some (.list (encode items)))
    (selected : items[position]? = some value) (env : HeapCalls.Env)
    (getValues : env "abund" = some (.ref address)) (getIndex : env indexName = some (number position)) :
    Evaluates program env (.subscript (.name "abund") (.name indexName)) heap (number value) heap := by
  apply Evaluates.subscript (by intro a b c impossible; cases impossible)
    (Evaluates.lookup getValues) (Evaluates.lookup getIndex)
  have item : (encode items)[position]? = some (number value) := by simp [encode, selected]
  have inside := (List.getElem?_eq_some_iff.mp item).1
  simpa only [(List.getElem?_eq_some_iff.mp item).2] using Mutable.getItem_list heap address position _ found inside

/-- The inner range marks its whole suffix, or skips a suffix whose sums all exceed the limit. -/
private theorem inner_loop (items remaining : List Nat) (ordered : remaining.Pairwise (· < ·))
    (limit a position address : Nat) (suffix : items.drop position = remaining)
    (base : Mutable.Heap) (found : base[address]? = some (.list (encode items)))
    (marked : Finset Nat) (env : HeapCalls.Env) (locals : Locals env limit items.length address base.length)
    (getA : env "a" = some (number a)) :
    ∃ finalEnv, Locals finalEnv limit items.length address base.length ∧
      (∀ key, key ≠ "j" → key ≠ "s" → finalEnv key = env key) ∧
      BlockContinues program env [.rangeStepLoop "j" position items.length 1 innerBody []]
        (workHeap base limit marked) finalEnv (workHeap base limit (marked ∪ row a remaining)) := by
  have addressBound := (List.getElem?_eq_some_iff.mp found).1
  induction remaining generalizing position marked env with
  | nil =>
      have finished : items.length ≤ position := List.drop_eq_nil_iff.mp suffix
      refine ⟨env, locals, by intros; rfl, ?_⟩
      simpa [row] using BlockContinues.range_done (module := program) (body := innerBody)
        (by simp [Calls.rangeContinues]; omega) BlockContinues.nil
  | cons head tail ih =>
      have selected : items[position]? = some head := by
        simpa [suffix] using (List.head?_drop (l := items) (i := position)).symm
      have inside := (List.getElem?_eq_some_iff.mp selected).1
      let boundEnv := setLocal env "j" (number position)
      let sumEnv := setLocal boundEnv "s" (number (a + head))
      have nextLocals : Locals sumEnv limit items.length address base.length :=
        ⟨by simpa [sumEnv, boundEnv, setLocal] using locals.bound,
         by simpa [sumEnv, boundEnv, setLocal] using locals.length,
         by simpa [sumEnv, boundEnv, setLocal] using locals.values,
         by simpa [sumEnv, boundEnv, setLocal] using locals.buffer⟩
      have preserved : ∀ key, key ≠ "j" → key ≠ "s" → sumEnv key = env key := by
        intro key notJ notS
        simp [sumEnv, boundEnv, setLocal, notJ, notS]
      have assigned : BlockContinues program boundEnv [.assign [.name "s"] (.binOp (.name "a") .add (.subscript (.name "abund") (.name "j")))]
          (workHeap base limit marked) sumEnv (workHeap base limit marked) := by
        apply BlockContinues.assign
        apply Evaluates.binary (av := number a) (bv := number head)
          (Evaluates.lookup (by simpa [boundEnv, setLocal] using getA))
          (read_item items position head address "j" (workHeap base limit marked)
            (by simpa only [List.getElem?_append_left addressBound] using found) selected boundEnv
            (by simpa [boundEnv, setLocal] using locals.values) rfl)
        simp [Mutable.binary, scalarBinary, number]
      have tested : Evaluates program sumEnv (.compare (.name "s") [(.gt, .name "lim")])
          (workHeap base limit marked) (.scalar (.bool (decide (limit < a + head)))) (workHeap base limit marked) := by
        apply Evaluates.compare (Evaluates.lookup rfl) (Evaluates.lookup nextLocals.bound)
        simp [HeapCalls.compare, scalarCompare, evalCmpOp]
        omega
      by_cases large : limit < a + head
      · have unchanged : flags (limit + 1) (marked ∪ row a (head :: tail)) = flags (limit + 1) marked := by
          apply flags_congr
          intro n bound
          simp [row_above a head limit tail ordered large n (by omega)]
        refine ⟨sumEnv, nextLocals, preserved, ?_⟩
        simp only [workHeap, unchanged]
        apply BlockContinues.range_step (by simp [Calls.rangeContinues]; omega)
        apply BlockContinues.append assigned
        apply BlockContinues.branch_next (flag := true) (by simpa [large] using tested) rfl
        exact BlockContinues.break_loop rfl BlockContinues.nil
      · have within : a + head < limit + 1 := by omega
        have written : BlockContinues program sumEnv [.assign [.subscript (.name "can") (.name "s")] (.constant (.int 1))]
            (workHeap base limit marked) sumEnv (workHeap base limit (insert (a + head) marked)) := by
          apply BlockContinues.setItem (by intro x y z impossible; cases impossible) Evaluates.constant
            (.cons (Evaluates.lookup nextLocals.buffer) (.cons (Evaluates.lookup rfl) .nil))
          have write := Mutable.setItem_buffer (workHeap base limit marked) base.length (a + head)
            (flags (limit + 1) marked) 1 (by simp [workHeap]) (by simpa using within)
          simpa [workHeap, flags_set _ _ _ within] using write
        obtain ⟨finalEnv, finalLocals, finalPreserved, rest⟩ := ih (List.pairwise_cons.mp ordered).2
          (position + 1) (by simp [List.drop_add_one_eq_tail_drop, suffix])
          (insert (a + head) marked) sumEnv nextLocals
          (by rw [preserved "a" (by decide) (by decide)]; exact getA)
        have same : insert (a + head) marked ∪ row a tail = marked ∪ row a (head :: tail) := by
          ext n
          simp [row]
        rw [same] at rest
        refine ⟨finalEnv, finalLocals, ?_, ?_⟩
        · intro key notJ notS
          exact (finalPreserved key notJ notS).trans (preserved key notJ notS)
        · apply BlockContinues.range_step (by simp [Calls.rangeContinues]; omega)
          apply BlockContinues.append assigned
          apply BlockContinues.branch_next (flag := false) (by simpa [large] using tested) rfl
          apply BlockContinues.append written
          exact BlockContinues.loop_resume rest

private def outerBody : List Stmt :=
  [.assign [.name "a"] (.subscript (.name "abund") (.name "i")),
   .forIn (.name "j") (.call (.name "range") [.name "i", .name "L"]) innerBody []]

/-- Each outer iteration adds one triangular row to the marked bytearray. -/
private theorem outer_loop (items remaining : List Nat) (ordered : remaining.Pairwise (· < ·))
    (limit position address : Nat) (suffix : items.drop position = remaining)
    (base : Mutable.Heap) (found : base[address]? = some (.list (encode items)))
    (marked : Finset Nat) (env : HeapCalls.Env) (locals : Locals env limit items.length address base.length) :
    ∃ finalEnv, Locals finalEnv limit items.length address base.length ∧
      (∀ key, key ≠ "i" → key ≠ "a" → key ≠ "j" → key ≠ "s" → finalEnv key = env key) ∧
      BlockContinues program env [.rangeStepLoop "i" position items.length 1 outerBody []]
        (workHeap base limit marked) finalEnv (workHeap base limit (marked ∪ pairs remaining)) := by
  have addressBound := (List.getElem?_eq_some_iff.mp found).1
  induction remaining generalizing position marked env with
  | nil =>
      have finished : items.length ≤ position := List.drop_eq_nil_iff.mp suffix
      refine ⟨env, locals, by intros; rfl, ?_⟩
      simpa [pairs] using BlockContinues.range_done (module := program) (body := outerBody)
        (by simp [Calls.rangeContinues]; omega) BlockContinues.nil
  | cons head tail ih =>
      have selected : items[position]? = some head := by
        simpa [suffix] using (List.head?_drop (l := items) (i := position)).symm
      have inside := (List.getElem?_eq_some_iff.mp selected).1
      let boundEnv := setLocal env "i" (number position)
      let firstEnv := setLocal boundEnv "a" (number head)
      have nextLocals : Locals firstEnv limit items.length address base.length :=
        ⟨by simpa [firstEnv, boundEnv, setLocal] using locals.bound,
         by simpa [firstEnv, boundEnv, setLocal] using locals.length,
         by simpa [firstEnv, boundEnv, setLocal] using locals.values,
         by simpa [firstEnv, boundEnv, setLocal] using locals.buffer⟩
      have assigned : BlockContinues program boundEnv
          [.assign [.name "a"] (.subscript (.name "abund") (.name "i"))]
          (workHeap base limit marked) firstEnv (workHeap base limit marked) :=
        BlockContinues.assign (read_item items position head address "i" (workHeap base limit marked)
          (by simpa only [List.getElem?_append_left addressBound] using found) selected boundEnv
          (by simpa [boundEnv, setLocal] using locals.values) rfl)
      obtain ⟨middleEnv, middleLocals, middlePreserved, inner⟩ := inner_loop items (head :: tail) ordered
        limit head position address suffix base found marked firstEnv nextLocals rfl
      have rowExecuted : BlockContinues program firstEnv
          [.forIn (.name "j") (.call (.name "range") [.name "i", .name "L"]) innerBody []]
          (workHeap base limit marked) middleEnv (workHeap base limit (marked ∪ row head (head :: tail))) := by
        apply BlockContinues.for_range
          (.cons (Evaluates.lookup rfl) (.cons (Evaluates.lookup nextLocals.length) .nil)) rfl
        exact inner
      obtain ⟨finalEnv, finalLocals, finalPreserved, rest⟩ := ih (List.pairwise_cons.mp ordered).2
        (position + 1) (by simp [List.drop_add_one_eq_tail_drop, suffix])
        (marked ∪ row head (head :: tail)) middleEnv middleLocals
      have same : (marked ∪ row head (head :: tail)) ∪ pairs tail = marked ∪ pairs (head :: tail) := by
        simp [pairs, Finset.union_assoc]
      rw [same] at rest
      refine ⟨finalEnv, finalLocals, ?_, ?_⟩
      · intro key notI notA notJ notS
        rw [finalPreserved key notI notA notJ notS, middlePreserved key notJ notS]
        simp [firstEnv, boundEnv, setLocal, notI, notA]
      · apply BlockContinues.range_step (by simp [Calls.rangeContinues]; omega)
        apply BlockContinues.append assigned
        apply BlockContinues.append rowExecuted
        simpa [Calls.loopContinuation, outerBody] using rest

/-- Running the actual source loops marks precisely the representable sums within the buffer. -/
theorem result (items : List Nat) (ordered : items.Pairwise (· < ·)) (limit address : Nat)
    (base : Mutable.Heap) (found : base[address]? = some (.list (encode items)))
    (env : HeapCalls.Env) (locals : Locals env limit items.length address base.length) :
    ∃ finalEnv, Locals finalEnv limit items.length address base.length ∧
      (∀ key, key ≠ "i" → key ≠ "a" → key ≠ "j" → key ≠ "s" → finalEnv key = env key) ∧
      BlockContinues program env [.forIn (.name "i") (.call (.name "range") [.name "L"]) outerBody []]
        (base ++ [.bytearray (List.replicate (limit + 1) 0)]) finalEnv
        (workHeap base limit (pairs items)) := by
  obtain ⟨finalEnv, finalLocals, preserved, executed⟩ := outer_loop items items ordered limit 0 address rfl
    base found ∅ env locals
  refine ⟨finalEnv, finalLocals, preserved, ?_⟩
  apply BlockContinues.for_range (.cons (Evaluates.lookup locals.length) .nil) rfl
  simpa [workHeap, flags] using executed

end Marking

namespace Summation
open ProjectEulerStatements.AbundantSums
private abbrev number (n : Nat) : Mutable.Value := .scalar (.int n)

theorem read_flag (limit position address : Nat) (inside : position ≤ limit) (marked : Finset Nat)
    (heap : Mutable.Heap) (found : heap[address]? = some (.bytearray (Marking.flags (limit + 1) marked)))
    (env : HeapCalls.Env) (getCan : env "can" = some (.ref address))
    (ordinary : ∀ a b c, expression ≠ .slice a b c)
    (index : Evaluates program env expression heap (number position) heap) :
    Evaluates program env (.subscript (.name "can") expression) heap
      (number (if position ∈ marked then 1 else 0)) heap := by
  apply Evaluates.subscript ordinary (Evaluates.lookup getCan) index
  have selected := Marking.flags_get (limit + 1) marked position (by omega)
  have read := Mutable.getItem_buffer heap address position (Marking.flags (limit + 1) marked)
    found (by simpa using (show position < limit + 1 by omega))
  rw [(List.getElem?_eq_some_iff.mp selected).2] at read
  by_cases present : position ∈ marked <;> simpa [present, number] using read

private def body : List Stmt :=
  [.ifThen (.unaryOp .not (.subscript (.name "can") (.name "n")))
    [.augAssign (.name "total") .add (.name "n")] []]

private theorem loop (limit count start total address : Nat) (stop : start + count = limit + 1)
    (marked : Finset Nat) (heap : Mutable.Heap)
    (found : heap[address]? = some (.bytearray (Marking.flags (limit + 1) marked)))
    (env : HeapCalls.Env) (getCan : env "can" = some (.ref address))
    (getTotal : env "total" = some (number total)) :
    BlockReturns program env
      [.rangeStepLoop "n" start (limit + 1) 1 body [], .return (.name "total")]
      heap (number (total + unmarkedSum marked start count)) heap := by
  induction count generalizing start total env with
  | zero =>
      have same : start = limit + 1 := by omega
      subst start
      apply BlockReturns.range_done (by simp [Calls.rangeContinues])
      simpa [unmarkedSum] using BlockReturns.return_value (Evaluates.lookup getTotal)
  | succ count ih =>
      let boundEnv := setLocal env "n" (number start)
      have savedCan : boundEnv "can" = some (.ref address) := by simpa [boundEnv, setLocal] using getCan
      have tested : Evaluates program boundEnv (.unaryOp .not (.subscript (.name "can") (.name "n")))
          heap (.scalar (.bool (decide (start ∉ marked)))) heap := by
        apply Evaluates.unary (read_flag limit start address (by omega) marked heap found boundEnv savedCan
          (by intro a b c impossible; cases impossible) (Evaluates.lookup rfl))
        by_cases present : start ∈ marked <;> simp [HeapCalls.unary, Mutable.truth, scalarTruth, present]
      apply BlockReturns.range_step (by simp [Calls.rangeContinues]; omega)
      by_cases missing : start ∉ marked
      · apply BlockReturns.branch (flag := true) (by simpa [missing] using tested) rfl
        apply BlockReturns.augAssign (left := number total) (right := number start)
          (value := number (total + start)) (middle := heap) (afterOperation := heap)
          (by simpa [boundEnv, setLocal] using getTotal) (Evaluates.lookup rfl)
        · simp [inplace, Mutable.binary, scalarBinary, number]
        · simpa [body, Calls.loopContinuation, Calls.loopControl, Calls.loopControlStmt,
            unmarkedSum, List.range'_succ, missing, Nat.add_assoc, Nat.add_left_comm, Nat.add_comm]
            using ih (start + 1) (total + start) (by omega) (setLocal boundEnv "total" (number (total + start)))
              (by simpa [setLocal] using savedCan) (by simp [setLocal])
      · apply BlockReturns.branch (flag := false) (by simpa [missing] using tested) rfl
        simpa [body, Calls.loopContinuation, Calls.loopControl, Calls.loopControlStmt,
            unmarkedSum, List.range'_succ, missing]
          using ih (start + 1) total (by omega) boundEnv savedCan
            (by simpa [boundEnv, setLocal] using getTotal)

def statements : List Stmt :=
  [.assign [.name "total"] (.constant (.int 0)),
   .forIn (.name "n") (.call (.name "range")
     [.constant (.int 1), .binOp (.name "lim") .add (.constant (.int 1))]) body [],
   .return (.name "total")]

theorem result (limit address : Nat) (marked : Finset Nat) (heap : Mutable.Heap)
    (found : heap[address]? = some (.bytearray (Marking.flags (limit + 1) marked)))
    (env : HeapCalls.Env) (getCan : env "can" = some (.ref address))
    (getLimit : env "lim" = some (number limit)) :
    BlockReturns program env statements heap (number (unmarkedSum marked 1 limit)) heap := by
  apply BlockReturns.assign Evaluates.constant
  apply BlockReturns.for_range
    (.cons Evaluates.constant (.cons (Evaluates.binary
      (Evaluates.lookup (by simpa [setLocal] using getLimit)) Evaluates.constant rfl) .nil)) rfl
  simpa only [Nat.zero_add] using loop limit limit 1 0 address (by omega) marked heap found
    (setLocal env "total" (number 0)) (by simpa [setLocal] using getCan) (by simp [setLocal])

end Summation

namespace Optimized
open ProjectEulerStatements.AbundantSums
private abbrev number (n : Nat) : Mutable.Value := .scalar (.int n)
private abbrev encode (items : List Nat) : List Mutable.Value := items.map number
private def definition : List Arg × List Stmt :=
  (findFunctionIn program.body "solve").getD ([], [])
private abbrev sieveHeap (limit : Nat) : Mutable.Heap :=
  [.list [number 0], .list (Sieve.table (limit + 1) ProjectEulerStatements.P21.d)]
private abbrev baseHeap (limit : Nat) : Mutable.Heap :=
  sieveHeap limit ++ [.list (encode (values limit))]
def finalHeap (limit : Nat) : Mutable.Heap :=
  baseHeap limit ++ [.bytearray (Marking.flags (limit + 1) (pairs (values limit)))]

/-- The optimized Python algorithm returns the statement's sum, including its embedded assertions. -/
theorem result (limit : Nat) (large : 28 ≤ limit) :
    ReturnsValue program "solve" [number limit] []
      (number (ProjectEulerStatements.P23.naive limit)) (finalHeap limit) := by
  let entry := functionEnv definition.1 definition.2 [number limit] []
  let sieveEnv := setLocal entry "sums" (.ref 1)
  let abundantEnv := setLocal sieveEnv "abundant" (.ref 2)
  let bufferEnv := setLocal abundantEnv "can" (.ref 3)
  let aliasEnv := setLocal bufferEnv "abund" (.ref 2)
  let limitEnv := setLocal aliasEnv "lim" (number limit)
  let allEnv := setLocal limitEnv "L" (number (values limit).length)
  let allocated := baseHeap limit ++ [.bytearray (List.replicate (limit + 1) 0)]
  have perfect : ProjectEulerStatements.P21.d 28 = 28 := twenty_eight_perfect
  have perfectTest : Evaluates program sieveEnv
      (.compare (.subscript (.name "sums") (.constant (.int 28))) [(.eq, .constant (.int 28))])
      (sieveHeap limit) (.scalar (.bool true)) (sieveHeap limit) := by
    apply Evaluates.compare (av := number 28) (bv := number 28)
      (middle := sieveHeap limit) (afterRight := sieveHeap limit)
    · apply Evaluates.subscript (by intro a b c impossible; cases impossible) (Evaluates.lookup rfl) Evaluates.constant
      have item := Sieve.table_lookup (limit + 1) ProjectEulerStatements.P21.d 28 (by omega)
      rw [perfect] at item
      simpa only [(List.getElem?_eq_some_iff.mp item).2] using
        Mutable.getItem_list (sieveHeap limit) 1 28 _ rfl (by simpa using (show 28 < limit + 1 by omega))
    · exact Evaluates.constant
    · rfl
  have first : (values limit).head? = some 12 := values_head limit (by omega)
  have nonempty : values limit ≠ [] := by
    intro empty
    simp [empty] at first
  have abundantTest : Evaluates program abundantEnv
      (.boolOp .and [.name "abundant", .compare (.subscript (.name "abundant") (.constant (.int 0)))
        [(.eq, .constant (.int 12))]]) (baseHeap limit) (.scalar (.bool true)) (baseHeap limit) := by
    apply Evaluates.boolean_next (value := .ref 2) (middle := baseHeap limit) (flag := true)
      (Evaluates.lookup rfl)
    · simp [Mutable.truth, baseHeap, sieveHeap, encode, nonempty]
    · rfl
    · apply Evaluates.boolean_last (flag := true)
      · apply Evaluates.compare (av := number 12) (bv := number 12)
          (middle := baseHeap limit) (afterRight := baseHeap limit)
        · apply Evaluates.subscript (by intro a b c impossible; cases impossible) (Evaluates.lookup rfl) Evaluates.constant
          have item : (encode (values limit))[0]? = some (number 12) := by
            have firstIndex : (values limit)[0]? = some 12 := by
              simpa only [List.head?_eq_getElem?] using first
            simp only [encode, List.getElem?_map, firstIndex, Option.map_some]
          simpa only [(List.getElem?_eq_some_iff.mp item).2] using
            Mutable.getItem_list (baseHeap limit) 2 0 _ rfl (List.getElem?_eq_some_iff.mp item).1
        · exact Evaluates.constant
        · rfl
      · rfl
  have allocate : Evaluates program abundantEnv
      (.builtinCall .bytearray [.binOp (.name "limit") .add (.constant (.int 1))])
      (baseHeap limit) (.ref 3) allocated := by
    apply Evaluates.builtin (.cons (Evaluates.binary (Evaluates.lookup rfl) Evaluates.constant rfl) .nil)
    simp [HeapCalls.primitive, Mutable.bytearray, Mutable.allocateBuffer, allocated, baseHeap, sieveHeap, Except.map,
      show ¬ (limit : Int) + 1 < 0 by omega]
  have locals : Marking.Locals allEnv limit (values limit).length 2 (baseHeap limit).length :=
    ⟨rfl, rfl, rfl, rfl⟩
  obtain ⟨finalEnv, finalLocals, _, marked⟩ := Marking.result (values limit) (values_sorted limit) limit 2
    (baseHeap limit) rfl allEnv locals
  have twelve : 12 ∈ values limit := by
    rw [mem_values]
    simp [ProjectEulerStatements.P23.abundantSet, twelve_abundant, show 12 ≤ limit by omega]
  have twentyFour : 24 ∈ pairs (values limit) := (mem_pairs 24 (values limit)).mpr
    ⟨12, twelve, 12, twelve, rfl⟩
  have flag24 : Evaluates program finalEnv (.subscript (.name "can") (.constant (.int 24)))
      (finalHeap limit) (number 1) (finalHeap limit) := by
    simpa [twentyFour] using Summation.read_flag limit 24 3 (by omega) (pairs (values limit))
      (finalHeap limit) rfl finalEnv finalLocals.buffer
      (by intro a b c impossible; cases impossible) Evaluates.constant
  apply BlockReturns.function (args := definition.1) (body := definition.2) rfl rfl
  apply BlockReturns.assign (value := .ref 1) (middle := sieveHeap limit)
  · exact Evaluates.call (.cons (Evaluates.lookup rfl) .nil) (Sieve.result limit [])
  · apply BlockReturns.assertion perfectTest rfl
    apply BlockReturns.assign (value := .ref 2) (middle := baseHeap limit)
    · exact AbundantList.result limit 1 (by omega) (sieveHeap limit) rfl sieveEnv rfl rfl
    · apply BlockReturns.assertion abundantTest rfl
      apply BlockReturns.assign allocate
      apply BlockReturns.assign (Evaluates.lookup rfl)
      apply BlockReturns.assign (Evaluates.lookup rfl)
      apply BlockReturns.assign (value := number (values limit).length) (middle := allocated)
      · apply Evaluates.builtin (.cons (Evaluates.lookup rfl) .nil)
        simp [HeapCalls.primitive, Mutable.length, allocated, sieveHeap, encode]
      · apply BlockContinues.returns marked
        apply BlockReturns.assertion (Evaluates.compare flag24 Evaluates.constant rfl) rfl
        have summed := Summation.result limit 3 (pairs (values limit)) (finalHeap limit) rfl finalEnv
          finalLocals.buffer finalLocals.bound
        simpa only [unmarkedSum_values] using summed

end Optimized

private abbrev inputEnv (limit : Int) : ScalarEnv :=
  fun key => if key = "limit" then some (.int limit) else none

private theorem accepted_limit (limit : Int)
    (requires : Calls.Condition program (inputEnv limit) Solvers.P23Ast.requires_solve_0) :
    28 ≤ limit := by
  simpa [evalCmpOp] using Calls.Evaluates.condition_int_compare requires rfl

theorem solve.«Equiv(naive)» (limit : Int)
    (requires : Calls.Condition program (inputEnv limit) Solvers.P23Ast.requires_solve_0) :
    (SetCalls.call program "solve" [.scalar (.int limit)] []).Equiv
      (SetCalls.call P23Ast.module "naive" [.scalar (.int limit)] []) := by
  have large := accepted_limit limit requires
  lift limit to Nat using (by omega : 0 ≤ limit)
  have large : 28 ≤ limit := by omega
  intro leftChoices rightChoices
  obtain ⟨referenceHeap, reference⟩ :=
    data.«project-euler-statements».data.py.project_euler_statements.p23.naive.result limit rightChoices
  apply Mutable.Call.equiv_of_finishes
    (a := (SetCalls.call program "solve" [.scalar (.int limit)] []).run leftChoices)
    (b := (SetCalls.call P23Ast.module "naive" [.scalar (.int limit)] []).run rightChoices)
    (SetCalls.finishes_of_heap (Optimized.result limit large).finishes leftChoices 0) reference
  exact Mutable.Completion.equivalent_allocations [] (Optimized.finalHeap limit) referenceHeap
    (.scalar (.int (ProjectEulerStatements.P23.naive limit))) [] (by simp [Mutable.Heap.Valid]) trivial (by simp)

theorem solve.«Supported» :
    (∀ (limit : Int), Calls.Condition program (inputEnv limit) Solvers.P23Ast.requires_solve_0 →
      (SetCalls.call program "solve" [.scalar (.int limit)] []).Supported) ∧
    (∀ (limit : Int), Calls.Condition program (inputEnv limit) Solvers.P23Ast.requires_solve_0 →
      (SetCalls.call P23Ast.module "naive" [.scalar (.int limit)] []).Supported) := by
  constructor
  · intro limit requires
    have large := accepted_limit limit requires
    lift limit to Nat using (by omega : 0 ≤ limit)
    have large : 28 ≤ limit := by omega
    intro choices
    exact (SetCalls.returns_of_heap (Optimized.result limit large).returns choices).supported
  · intro limit _
    exact data.«project-euler-statements».data.py.project_euler_statements.p23.naive.«Supported» limit

end solvers.«23»
