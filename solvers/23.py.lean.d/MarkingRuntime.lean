import solvers.«23.py.lean.d».AbundantRuntime
import lang.py.FormalSW.HeapRuntimeCostLoopControl

/-! The actual triangular pair loop, including its early `break`, has a
quadratic bound independent of the values stored in the bytearray. -/
namespace solvers.«23».runtimeCost.Marking
open FormalSW.PyAst HeapCalls HeapCalls.Runtime
abbrev program := Solvers.P23Ast.module
abbrev number (n : Nat) : Mutable.Value := .scalar (.int n)
abbrev encode := Abundant.encode

structure Locals (env : HeapCalls.Env) (limit length address buffer : Nat) : Prop where
  bound : env "lim" = some (number limit)
  length : env "L" = some (number length)
  values : env "abund" = some (.ref address)
  buffer : env "can" = some (.ref buffer)

def innerBody : List Stmt :=
  [.assign [.name "s"] (.binOp (.name "a") .add (.subscript (.name "abund") (.name "j"))),
   .ifThen (.compare (.name "s") [(.gt, .name "lim")]) [.breakLoop] [],
   .assign [.subscript (.name "can") (.name "s")] (.constant (.int 1))]

def outerBody : List Stmt :=
  [.assign [.name "a"] (.subscript (.name "abund") (.name "i")),
   .forIn (.name "j") (.call (.name "range") [.name "i", .name "L"]) innerBody []]

def statements : List Stmt :=
  [.forIn (.name "i") (.call (.name "range") [.name "L"]) outerBody []]

theorem read_item (items : List Nat) (position value address : Nat) (indexName : String)
    (heap : Mutable.Heap) (found : heap[address]? = some (.list (encode items)))
    (selected : items[position]? = some value) (env : HeapCalls.Env)
    (getValues : env "abund" = some (.ref address))
    (getIndex : env indexName = some (number position)) :
    EvaluatesIn .word program env (.subscript (.name "abund") (.name indexName)) heap 10
      (number value) heap := by
  apply EvaluatesIn.list_subscript (by intro a b c equal; cases equal)
    (EvaluatesIn.lookup getValues) (EvaluatesIn.lookup getIndex) found
  have item : (encode items)[position]? = some (number value) := by
    simp [encode, Abundant.encode, selected]
  simpa only [(List.getElem?_eq_some_iff.mp item).2] using
    Mutable.getItem_list heap address position _ found (List.getElem?_eq_some_iff.mp item).1

/-- Exit the current source loop without consuming the caller's continuation. -/
private theorem break_tail {body : List Stmt} {next : Stmt}
    (target : ∀ following, Calls.loopTarget (body ++ following) = some (next, following))
    (env : HeapCalls.Env) (heap : Mutable.Heap) :
    BlockContinuesIn .word program env (.breakLoop :: body) heap 1 env heap := by
  intro following anchors stack
  exact advance (by simp [step, target]) rfl rfl

theorem inner_loop (items remaining : List Nat) (limit a position address : Nat)
    (suffix : items.drop position = remaining)
    (base : Mutable.Heap) (found : base[address]? = some (.list (encode items)))
    (bytes : List UInt8) (size : bytes.length = limit + 1)
    (env : HeapCalls.Env) (locals : Locals env limit items.length address base.length)
    (getA : env "a" = some (number a)) :
    ∃ finalBytes finalEnv, finalBytes.length = limit + 1 ∧
      Locals finalEnv limit items.length address base.length ∧
      (∀ key, key ≠ "j" → key ≠ "s" → finalEnv key = env key) ∧
      BlockContinuesIn .word program env
        [.rangeStepLoop "j" position items.length 1 innerBody []]
        (base ++ [.bytearray bytes]) (50 * remaining.length + 4) finalEnv
        (base ++ [.bytearray finalBytes]) := by
  have addressBound := (List.getElem?_eq_some_iff.mp found).1
  induction remaining generalizing position bytes env with
  | nil =>
      have finished : items.length ≤ position := List.drop_eq_nil_iff.mp suffix
      refine ⟨bytes, env, size, locals, by intros; rfl, ?_⟩
      exact BlockContinuesIn.range_done (by simp [Calls.rangeContinues]; omega) BlockContinuesIn.nil
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
      have assigned : BlockContinuesIn .word program boundEnv
          [.assign [.name "s"] (.binOp (.name "a") .add (.subscript (.name "abund") (.name "j")))]
          (base ++ [.bytearray bytes]) 17 sumEnv (base ++ [.bytearray bytes]) := by
        apply BlockContinuesIn.assign
        refine EvaluatesIn.binary (op := .add) (av := .int a) (bv := .int head)
          (value := number (a + head)) (ca := 1) (cb := 10) (work := 1)
          (heap := base ++ [Mutable.Object.bytearray bytes])
          (middle := base ++ [Mutable.Object.bytearray bytes])
          (afterRight := base ++ [Mutable.Object.bytearray bytes]) ?_ ?_ ?_ ?_
        · exact EvaluatesIn.lookup (name := "a") (by simpa [boundEnv, setLocal] using getA)
        · exact read_item items position head address "j" (base ++ [.bytearray bytes])
            (by simpa only [List.getElem?_append_left addressBound] using found) selected boundEnv
            (by simpa [boundEnv, setLocal] using locals.values) rfl
        · simp [Mutable.binary, scalarBinary, number]
        · rfl
      have tested : EvaluatesIn .word program sumEnv (.compare (.name "s") [(.gt, .name "lim")])
          (base ++ [.bytearray bytes]) 7 (.scalar (.bool (decide (limit < a + head))))
          (base ++ [.bytearray bytes]) := by
        apply EvaluatesIn.compare (work := 2) (EvaluatesIn.lookup (name := "s") rfl)
          (EvaluatesIn.lookup nextLocals.bound)
        · simp [HeapCalls.compare, scalarCompare, evalCmpOp]
          omega
        · rfl
      by_cases large : limit < a + head
      · refine ⟨bytes, sumEnv, size, nextLocals, preserved, ?_⟩
        have stop : BlockContinuesIn .word program sumEnv
            [.breakLoop, .assign [.subscript (.name "can") (.name "s")] (.constant (.int 1)),
             .loopResume (.rangeStepLoop "j" ((position : Int) + 1) items.length 1 innerBody [])]
            (base ++ [.bytearray bytes]) 1 sumEnv (base ++ [.bytearray bytes]) :=
          break_tail (by intro following; rfl) sumEnv _
        have branch := BlockContinuesIn.branch_tail (yes := [.breakLoop]) (no := [])
          (flag := true) (by simpa [large] using tested) rfl stop
        have run := BlockContinuesIn.range_step (env := env) (name := "j")
          (next := (position : Int)) (stop := (items.length : Int)) (stride := 1)
          (body := innerBody) (otherwise := [])
          (by simp [Calls.rangeContinues]; omega)
          (by simpa [innerBody, Calls.loopContinuation, Calls.loopControl, Calls.loopControlStmt,
            boundEnv, number] using assigned.append branch)
        exact run.mono (by simp; omega)
      · have within : a + head < bytes.length := by omega
        let nextBytes := bytes.set (a + head) 1
        have written : BlockContinuesIn .word program sumEnv
            [.assign [.subscript (.name "can") (.name "s")] (.constant (.int 1))]
            (base ++ [.bytearray bytes]) 14 sumEnv (base ++ [.bytearray nextBytes]) := by
          have bufferFound : (base ++ [.bytearray bytes])[base.length]? = some (.bytearray bytes) := by simp
          have write := BlockContinuesIn.buffer_setItem (globals := [])
            (by intro a b c equal; cases equal)
            (EvaluatesIn.constant (arithmetic := .word) (module := program)
              (env := sumEnv) (heap := base ++ [.bytearray bytes]) (value := .int 1))
            (ArgumentsIn.cons (EvaluatesIn.lookup nextLocals.buffer)
              (ArgumentsIn.cons (EvaluatesIn.lookup (name := "s") rfl) ArgumentsIn.nil))
            bufferFound (Mutable.setItem_buffer _ base.length (a + head) bytes 1 bufferFound within)
          simpa [nextBytes, List.set_append_right, PrimitiveWork.integer] using write
        obtain ⟨finalBytes, finalEnv, finalSize, finalLocals, finalPreserved, rest⟩ :=
          ih (position + 1) (by simp [List.drop_add_one_eq_tail_drop, suffix])
            nextBytes (by simpa [nextBytes] using size) sumEnv nextLocals
            (by rw [preserved "a" (by decide) (by decide)]; exact getA)
        refine ⟨finalBytes, finalEnv, finalSize, finalLocals, ?_, ?_⟩
        · intro key notJ notS
          exact (finalPreserved key notJ notS).trans (preserved key notJ notS)
        · have resume := BlockContinuesIn.loop_resume rest
          have branch := BlockContinuesIn.branch_tail (yes := [.breakLoop]) (no := [])
            (flag := false) (by simpa [large] using tested) rfl (written.append resume)
          have run := BlockContinuesIn.range_step (env := env) (name := "j")
            (next := (position : Int)) (stop := (items.length : Int)) (stride := 1)
            (body := innerBody) (otherwise := [])
            (by simp [Calls.rangeContinues]; omega)
            (by simpa [innerBody, Calls.loopContinuation, Calls.loopControl, Calls.loopControlStmt,
              boundEnv, number] using assigned.append branch)
          exact run.mono (by simp; omega)

theorem outer_loop (items remaining : List Nat) (limit position address : Nat)
    (suffix : items.drop position = remaining)
    (base : Mutable.Heap) (found : base[address]? = some (.list (encode items)))
    (bytes : List UInt8) (size : bytes.length = limit + 1)
    (env : HeapCalls.Env) (locals : Locals env limit items.length address base.length) :
    ∃ finalBytes finalEnv, finalBytes.length = limit + 1 ∧
      Locals finalEnv limit items.length address base.length ∧
      (∀ key, key ≠ "i" → key ≠ "a" → key ≠ "j" → key ≠ "s" → finalEnv key = env key) ∧
      BlockContinuesIn .word program env
        [.rangeStepLoop "i" position items.length 1 outerBody []]
        (base ++ [.bytearray bytes]) (60 * (items.length + 1) * remaining.length + 4) finalEnv
        (base ++ [.bytearray finalBytes]) := by
  have addressBound := (List.getElem?_eq_some_iff.mp found).1
  induction remaining generalizing position bytes env with
  | nil =>
      have finished : items.length ≤ position := List.drop_eq_nil_iff.mp suffix
      refine ⟨bytes, env, size, locals, by intros; rfl, ?_⟩
      exact BlockContinuesIn.range_done (by simp [Calls.rangeContinues]; omega) BlockContinuesIn.nil
  | cons head tail ih =>
      have selected : items[position]? = some head := by
        simpa [suffix] using (List.head?_drop (l := items) (i := position)).symm
      have inside := (List.getElem?_eq_some_iff.mp selected).1
      have suffixSize : (head :: tail).length ≤ items.length := by
        rw [← suffix, List.length_drop]
        omega
      let boundEnv := setLocal env "i" (number position)
      let firstEnv := setLocal boundEnv "a" (number head)
      have nextLocals : Locals firstEnv limit items.length address base.length :=
        ⟨by simpa [firstEnv, boundEnv, setLocal] using locals.bound,
         by simpa [firstEnv, boundEnv, setLocal] using locals.length,
         by simpa [firstEnv, boundEnv, setLocal] using locals.values,
         by simpa [firstEnv, boundEnv, setLocal] using locals.buffer⟩
      have assigned : BlockContinuesIn .word program boundEnv
          [.assign [.name "a"] (.subscript (.name "abund") (.name "i"))]
          (base ++ [.bytearray bytes]) 12 firstEnv (base ++ [.bytearray bytes]) :=
        BlockContinuesIn.assign (read_item items position head address "i" (base ++ [.bytearray bytes])
          (by simpa only [List.getElem?_append_left addressBound] using found) selected boundEnv
          (by simpa [boundEnv, setLocal] using locals.values) rfl)
      obtain ⟨middleBytes, middleEnv, middleSize, middleLocals, middlePreserved, inner⟩ :=
        inner_loop items (head :: tail) limit head position address suffix base found bytes size
          firstEnv nextLocals rfl
      have row := BlockContinuesIn.for_range_pair
        (EvaluatesIn.lookup (arithmetic := .word) (module := program) (env := firstEnv)
          (heap := base ++ [.bytearray bytes]) (name := "i") rfl)
        (EvaluatesIn.lookup nextLocals.length) inner
      obtain ⟨finalBytes, finalEnv, finalSize, finalLocals, finalPreserved, rest⟩ :=
        ih (position + 1) (by simp [List.drop_add_one_eq_tail_drop, suffix])
          middleBytes middleSize middleEnv middleLocals
      refine ⟨finalBytes, finalEnv, finalSize, finalLocals, ?_, ?_⟩
      · intro key notI notA notJ notS
        rw [finalPreserved key notI notA notJ notS, middlePreserved key notJ notS]
        simp [firstEnv, boundEnv, setLocal, notI, notA]
      · have run := BlockContinuesIn.range_step (env := env) (name := "i")
          (next := (position : Int)) (stop := (items.length : Int)) (stride := 1)
          (body := outerBody) (otherwise := [])
          (by simp [Calls.rangeContinues]; omega)
          (by simpa [outerBody, Calls.loopContinuation, boundEnv, number] using
            assigned.append (row.append rest))
        apply run.mono
        simp only [List.length_cons] at suffixSize
        simp only [List.length_cons, Nat.mul_add, Nat.mul_one]
        omega

theorem source_run (items : List Nat) (limit address : Nat)
    (base : Mutable.Heap) (found : base[address]? = some (.list (encode items)))
    (bytes : List UInt8) (size : bytes.length = limit + 1)
    (env : HeapCalls.Env) (locals : Locals env limit items.length address base.length) :
    ∃ finalBytes finalEnv, finalBytes.length = limit + 1 ∧
      Locals finalEnv limit items.length address base.length ∧
      BlockContinuesIn .word program env statements (base ++ [.bytearray bytes])
        (60 * (items.length + 1)^2 + 9) finalEnv (base ++ [.bytearray finalBytes]) := by
  obtain ⟨finalBytes, finalEnv, finalSize, finalLocals, _, run⟩ :=
    outer_loop items items limit 0 address (by simp) base found bytes size env locals
  refine ⟨finalBytes, finalEnv, finalSize, finalLocals, ?_⟩
  have ranged := BlockContinuesIn.for_range_bound (EvaluatesIn.lookup locals.length) run
  apply ranged.mono
  nlinarith

end solvers.«23».runtimeCost.Marking
