import «.formalsw».solvers.«21.py.ast»
import lang.py.FormalSW.HeapRuntimeCostListAugAssign
import lang.py.FormalSW.HeapRuntimeCostRanges

/-! Runtime of the actual inner multiple-adding loop. The invariant records only
integer entries and the table length; correctness of their divisor sums is not
needed to bound this loop's work. -/
namespace solvers.«21».runtimeCost.Sieve
open FormalSW.PyAst HeapCalls HeapCalls.Runtime

abbrev program := Solvers.P21Ast.module

def definition : List Arg × List Stmt :=
  (findFunctionIn program.body "proper_divisor_sums_sieve").getD ([], [])

def outerBody : List Stmt :=
  match ((executableBody definition.2)[1]? : Option Stmt) with
  | some (.forIn _ _ body _) => body
  | _ => []

def innerBody : List Stmt :=
  match (outerBody[0]? : Option Stmt) with
  | some (.forIn _ _ body _) => body
  | _ => []

theorem innerBody_eq : innerBody =
    [.augAssign (.subscript (.name "s") (.name "j")) .add (.name "i")] := rfl

theorem innerBody_continuation (next : Stmt) : Calls.loopContinuation innerBody next = next := rfl

def number (value : Int) : Mutable.Value := .scalar (.int value)
def encode (values : List Int) : List Mutable.Value := values.map number

@[simp] theorem encode_length (values : List Int) : (encode values).length = values.length := by
  simp [encode]

/-- One source `s[j] += i`, including receiver/index reads and the final write.
The integer entries need not be nonnegative or bounded in the word model. -/
theorem increment (heap : Heap) (env : HeapCalls.Env) (address i j : Nat) (values : List Int)
    (inside : j < values.length)
    (array : env "s" = some (.ref address))
    (divisor : env "i" = some (number i))
    (index : env "j" = some (number j))
    (found : heap[address]? = some (.list (encode values))) :
    BlockContinuesIn .word program env innerBody heap 16 env
      (heap.set address (.list (encode (values.set j (values[j] + (i : Int)))))) := by
  have written := BlockContinuesIn.list_augAssignItem (globals := [])
    (arithmetic := .word) (module := program) (env := env)
    (base := .name "s") (index := .name "j") (expression := .name "i")
    (op := .add) (cb := 1) (ci := 1) (cr := 1) (work := 1)
    (heap := heap) (afterReceiver := heap) (afterIndex := heap) (afterRhs := heap)
    (address := address) (offset := j) (items := encode values) (current := encode values)
    (left := .int values[j]) (right := .int i) (updated := number (values[j] + (i : Int)))
    (position := j)
    (by intro a b c equal; cases equal)
    (EvaluatesIn.lookup array) (EvaluatesIn.lookup index) found
    (by
      simpa only [encode, List.getElem_map, number] using
        Mutable.getItem_list heap address j (encode values) found (by simpa using inside))
    (EvaluatesIn.lookup divisor)
    (by simp [inplace, Mutable.binary, scalarBinary, number])
    rfl found
    (by
      have nonnegative : ¬ (j : Int) < 0 := by omega
      have bounded : ¬ values.length ≤ j := by omega
      simp [Mutable.setItem, Mutable.index, found, nonnegative, bounded, Except.map, number])
  have encoded : (encode values).set j (number (values[j] + (i : Int))) =
      encode (values.set j (values[j] + (i : Int))) := by
    simp [encode, List.map_set]
  simpa only [innerBody_eq, encoded] using written

private theorem set_existing (heap : Heap) (address : Nat) (object : Mutable.Object)
    (found : heap[address]? = some object) : heap.set address object = heap := by
  obtain ⟨bounded, equal⟩ := List.getElem?_eq_some_iff.mp found
  rw [← equal]
  exact List.set_getElem_self bounded

/-- `count` is any upper bound on the number of remaining range iterations:
`maximum + 1 ≤ j + count*i`. Choosing the exact ceiling gives a linear bound
in the actual iteration count. The heap outside the table is unchanged, every
entry remains an integer, and only local `j` may change. -/
theorem inner_loop (count maximum i j : Nat) (positive : 0 < i)
    (covers : maximum + 1 ≤ j + count * i)
    (heap : Heap) (env : HeapCalls.Env) (address : Nat) (values : List Int)
    (size : values.length = maximum + 1)
    (array : env "s" = some (.ref address))
    (divisor : env "i" = some (number i))
    (found : heap[address]? = some (.list (encode values))) :
    ∃ (finalValues : List Int) (finalEnv : HeapCalls.Env),
      finalValues.length = values.length ∧
      (∀ name, name ≠ "j" → finalEnv name = env name) ∧
      BlockContinuesIn .word program env
        [.rangeStepLoop "j" j (maximum + 1) i innerBody []] heap (20 * count + 4) finalEnv
        (heap.set address (.list (encode finalValues))) := by
  induction count generalizing j heap env values with
  | zero =>
      refine ⟨values, env, rfl, fun _ _ => rfl, ?_⟩
      rw [set_existing heap address _ found]
      apply BlockContinuesIn.range_done (budget := 0) _ BlockContinuesIn.nil
      simp [Calls.rangeContinues, positive] <;> omega
  | succ count ih =>
      by_cases active : j < maximum + 1
      · have inside : j < values.length := by omega
        let nextEnv := setLocal env "j" (number j)
        let nextValues := values.set j (values[j] + (i : Int))
        let nextHeap := heap.set address (.list (encode nextValues))
        have nextSize : nextValues.length = maximum + 1 := by
          simpa [nextValues] using size
        have nextArray : nextEnv "s" = some (.ref address) := by
          simpa [nextEnv, setLocal] using array
        have nextDivisor : nextEnv "i" = some (number i) := by
          simpa [nextEnv, setLocal] using divisor
        have addressBound : address < heap.length := (List.getElem?_eq_some_iff.mp found).1
        have nextFound : nextHeap[address]? = some (.list (encode nextValues)) := by
          exact List.getElem?_set_self addressBound
        have nextCovers : maximum + 1 ≤ (j + i) + count * i := by
          simpa only [Nat.succ_mul, Nat.add_assoc, Nat.add_comm, Nat.add_left_comm] using covers
        obtain ⟨finalValues, finalEnv, finalSize, preserved, rest⟩ :=
          ih (j + i) nextCovers nextHeap nextEnv nextValues nextSize nextArray nextDivisor nextFound
        refine ⟨finalValues, finalEnv, ?_, ?_, ?_⟩
        · simpa [nextValues] using finalSize
        · intro name different
          rw [preserved name different]
          simp [nextEnv, setLocal, different]
        · have one := increment heap nextEnv address i j values inside nextArray nextDivisor
            (by simp [nextEnv, setLocal]) found
          have remainder : BlockContinuesIn .word program nextEnv
              [.rangeStepLoop "j" ((j : Int) + (i : Int)) (maximum + 1) i innerBody []]
              nextHeap (20 * count + 4) finalEnv
              (heap.set address (.list (encode finalValues))) := by
            simpa only [Int.natCast_add, nextHeap, List.set_set] using rest
          have body := BlockContinuesIn.append one remainder
          have run := BlockContinuesIn.range_step (env := env) (name := "j")
            (next := (j : Int)) (stop := (maximum + 1 : Nat)) (stride := (i : Int))
            (body := innerBody) (otherwise := [])
            (by simp [Calls.rangeContinues, positive]; omega)
            (by simpa only [innerBody_continuation, nextEnv, number] using body)
          exact run.mono (by omega)
      · refine ⟨values, env, rfl, fun _ _ => rfl, ?_⟩
        rw [set_existing heap address _ found]
        exact (BlockContinuesIn.range_done (module := program) (env := env)
          (name := "j") (next := (j : Int)) (stop := (maximum + 1 : Nat)) (stride := (i : Int))
          (body := innerBody) (otherwise := []) (heap := heap)
          (by simp [Calls.rangeContinues, positive]; omega) BlockContinuesIn.nil).mono (by omega)

/-- Include the source `range(i * 2, n_max + 1, i)` evaluations and entry.
The additional 24 units comprise its two arithmetic bounds, divisor lookup,
three argument frames, block entry and range initialization. -/
theorem divisor_pass (count maximum i : Nat) (positive : 0 < i)
    (covers : maximum + 1 ≤ 2 * i + count * i)
    (heap : Heap) (env : HeapCalls.Env) (address : Nat) (values : List Int)
    (size : values.length = maximum + 1)
    (bound : env "n_max" = some (number maximum))
    (array : env "s" = some (.ref address))
    (divisor : env "i" = some (number i))
    (found : heap[address]? = some (.list (encode values))) :
    ∃ (finalValues : List Int) (finalEnv : HeapCalls.Env),
      finalValues.length = values.length ∧
      (∀ name, name ≠ "j" → finalEnv name = env name) ∧
      BlockContinuesIn .word program env outerBody heap (20 * count + 28) finalEnv
        (heap.set address (.list (encode finalValues))) := by
  obtain ⟨finalValues, finalEnv, finalSize, preserved, loop⟩ :=
    inner_loop count maximum i (2 * i) positive covers heap env address values size array divisor found
  refine ⟨finalValues, finalEnv, finalSize, preserved, ?_⟩
  have start : EvaluatesIn .word program env
      (.binOp (.name "i") .mult (.constant (.int 2))) heap 6 (number (2 * i)) heap := by
    apply EvaluatesIn.binary (work := 1) (EvaluatesIn.lookup divisor) EvaluatesIn.constant
    · simp [Mutable.binary, scalarBinary, number, Int.mul_comm]
    · rfl
  have stop : EvaluatesIn .word program env
      (.binOp (.name "n_max") .add (.constant (.int 1))) heap 6 (number (maximum + 1)) heap := by
    apply EvaluatesIn.binary (work := 1) (EvaluatesIn.lookup bound) EvaluatesIn.constant
    · simp [Mutable.binary, scalarBinary, number]
    · rfl
  have stride : EvaluatesIn .word program env (.name "i") heap 1 (number i) heap :=
    EvaluatesIn.lookup divisor
  intro following anchors stack
  let a := Expr.binOp (.name "i") .mult (.constant (.int 2))
  let b := Expr.binOp (.name "n_max") .add (.constant (.int 1))
  let c := Expr.name "i"
  have first : HeapCalls.Runtime.Reaches .word program anchors
      ⟨.block (outerBody ++ following) env, stack, heap⟩ 1
      ⟨.expr a env, .rangeArguments "j" [] [b, c] innerBody [] following env :: stack, heap⟩ :=
    advance rfl rfl rfl
  have second : HeapCalls.Runtime.Reaches .word program anchors
      ⟨.value (number (2 * i)), .rangeArguments "j" [] [b, c] innerBody [] following env :: stack, heap⟩ 1
      ⟨.expr b env, .rangeArguments "j" [number (2 * i)] [c] innerBody [] following env :: stack, heap⟩ :=
    advance rfl rfl rfl
  have third : HeapCalls.Runtime.Reaches .word program anchors
      ⟨.value (number (maximum + 1)), .rangeArguments "j" [number (2 * i)] [c] innerBody [] following env :: stack, heap⟩ 2
      ⟨.expr c env, .rangeArguments "j" [number (2 * i), number (maximum + 1)] [] innerBody [] following env :: stack, heap⟩ :=
    advance rfl rfl rfl
  have ready : HeapCalls.Runtime.Reaches .word program anchors
      ⟨.value (number i), .rangeArguments "j" [number (2 * i), number (maximum + 1)] [] innerBody [] following env :: stack, heap⟩ 3
      ⟨.startRange "j" [number (2 * i), number (maximum + 1), number i] innerBody [] following env, stack, heap⟩ :=
    advance rfl rfl rfl
  have enter : HeapCalls.Runtime.Reaches .word program anchors
      ⟨.startRange "j" [number (2 * i), number (maximum + 1), number i] innerBody [] following env, stack, heap⟩ 4
      ⟨.block ([.rangeStepLoop "j" (2 * i) (maximum + 1) i innerBody []] ++ following) env, stack, heap⟩ := by
    have bounds : rangeBounds [number (2 * i), number (maximum + 1), number i] =
        .ok (↑(2 * i), ↑(maximum + 1), (i : Int)) := by
      change (if (i : Int) = 0 then Except.error ({ typeName := "ValueError", args := [] } : ExceptionObject)
        else Except.ok (2 * (i : Int), (maximum : Int) + 1, (i : Int))) = _
      simp [show i ≠ 0 by omega]
    apply advance
    · simpa only [Int.natCast_add, Int.natCast_mul] using
        (show step program anchors
          ⟨.startRange "j" [number (2 * i), number (maximum + 1), number i] innerBody [] following env, stack, heap⟩ =
          .next ⟨.block ([.rangeStepLoop "j" (↑(2 * i)) (↑(maximum + 1)) i innerBody []] ++ following) env, stack, heap⟩ by
          simp [step, bounds])
    · rfl
    · rfl
  exact ((((((((first.trans (start anchors _)).trans second).trans (stop anchors _)).trans third).trans
    (stride anchors _)).trans ready).trans enter).trans (loop following anchors stack)).mono (by omega)

end solvers.«21».runtimeCost.Sieve
