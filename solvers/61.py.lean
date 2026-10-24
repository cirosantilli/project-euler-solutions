import «.formalsw».solvers.«61.py.ast»
import data.«project-euler-statements».data.py.project_euler_statements.«p61.py»
import FormalSW

namespace solvers.«61».polygonal
open FormalSW.PyAst

private abbrev answer :=
  data.«project-euler-statements».data.py.project_euler_statements.p61.polygonal.answer

/-- The actual helper preserves any surrounding loop's heap and global context. -/
theorem heap_result (s n : Int) (heap : Mutable.Heap) (globals : Mutable.Globals := []) :
    HeapCalls.ReturnsValue Solvers.P61Ast.module "polygonal" [.scalar (.int s), .scalar (.int n)] heap
      (.scalar (.int (answer s n))) heap globals := by
  let params := ((findFunctionIn Solvers.P61Ast.module.body "polygonal").getD ([], [])).1
  let body := ((findFunctionIn Solvers.P61Ast.module.body "polygonal").getD ([], [])).2
  let env := HeapCalls.functionEnv params body [.scalar (.int s), .scalar (.int n)] globals
  apply HeapCalls.BlockReturns.function rfl rfl
  change HeapCalls.BlockReturns Solvers.P61Ast.module env
    [.return (data.«project-euler-statements».data.py.project_euler_statements.p61.polygonal.expression "s" "n")]
    heap (.scalar (.int (answer s n))) heap globals
  apply HeapCalls.BlockReturns.return_value
  exact data.«project-euler-statements».data.py.project_euler_statements.p61.polygonal.expression_evaluates
    _ _ _ _ _ _ _ _ (by cases globals <;> rfl) (by cases globals <;> rfl)

/-- Source-level arithmetic, including Python's signed floor division. -/
theorem result (s n : Int) :
    Finishes (scalarFunction Solvers.P61Ast.module "polygonal" [.int s, .int n]).step
      (scalarFunction Solvers.P61Ast.module "polygonal" [.int s, .int n]).initial
      (.ok (.int (answer s n))) := by
  apply Execution.finishes_of_observe (fuel := 1)
  rfl

theorem «Returns» (s n : Int) :
    (scalarFunction Solvers.P61Ast.module "polygonal" [.int s, .int n]).Returns :=
  ⟨.int (answer s n), result s n⟩

theorem «Equiv(reference_polygonal)» (s n : Int) :
    (scalarFunction Solvers.P61Ast.module "polygonal" [.int s, .int n]).Equiv
      (scalarFunction P61Ast.module "polygonal" [.int s, .int n]) := by
  have reference :=
    data.«project-euler-statements».data.py.project_euler_statements.p61.polygonal.result s n
  have optimized := result s n
  intro outcome
  constructor
  · intro finished
    rw [finished.unique optimized]
    exact reference
  · intro finished
    rw [finished.unique reference]
    exact optimized

theorem «Supported» :
    (∀ s n : Int, (scalarFunction Solvers.P61Ast.module "polygonal" [.int s, .int n]).Supported) ∧
      (∀ s n : Int, (scalarFunction P61Ast.module "polygonal" [.int s, .int n]).Supported) :=
  ⟨fun s n => («Returns» s n).supported,
    data.«project-euler-statements».data.py.project_euler_statements.p61.polygonal.«Supported»⟩

end solvers.«61».polygonal

namespace solvers.«61».generate_4digit_polygonals
open FormalSW.PyAst FormalSW.PyAst.HeapCalls

private abbrev program := Solvers.P61Ast.module
private abbrev number (value : Int) : Mutable.Value := .scalar (.int value)
private abbrev flag (value : Bool) : Mutable.Value := .scalar (.bool value)
private abbrev polygonal :=
  data.«project-euler-statements».data.py.project_euler_statements.p61.polygonal.answer

def accepted (value : Int) : Prop := 1000 ≤ value ∧ value ≤ 9999 ∧ 10 ≤ value % 100

instance (value : Int) : Decidable (accepted value) := inferInstanceAs (Decidable (_ ∧ _ ∧ _))

def emitted (value : Int) : List Int := if accepted value then [value] else []

/-- The bound follows from polygonal growth, rather than restricting the source input. -/
def collect : Nat → Int → Int → List Int
  | 0, _, _ => []
  | fuel + 1, sides, index =>
      let value := polygonal sides index
      if 10000 ≤ value then [] else emitted value ++ collect fuel sides (index + 1)

def answer (sides : Int) : List Int := collect 140 sides 1

/-- Later positive indices never decrease within a polygonal family. -/
theorem monotone_from (s n : Int) (sides : 3 ≤ s) (positive : 1 ≤ n) (offset : Nat) :
    polygonal s n ≤ polygonal s (n + offset) := by
  induction offset with
  | zero => simp
  | succ offset ih =>
      have step := data.«project-euler-statements».data.py.project_euler_statements.p61.polygonal.increasing_step
        s (n + offset) sides (by omega)
      exact ih.trans (by simpa [Nat.cast_add, Nat.cast_one, add_assoc] using step.le)

/-- Early termination discards only values which fail the four-digit filter. -/
theorem collect_eq_window (fuel : Nat) (s n : Int) (sides : 3 ≤ s) (positive : 1 ≤ n) :
    collect fuel s n =
      ((List.range fuel).map (fun offset : Nat => polygonal s (n + offset))).filter
        (fun value => decide (accepted value)) := by
  induction fuel generalizing n with
  | zero => rfl
  | succ fuel ih =>
      have mapped : (List.range (fuel + 1)).map (fun offset : Nat => polygonal s (n + offset)) =
          polygonal s n :: (List.range fuel).map (fun offset : Nat => polygonal s (n + 1 + offset)) := by
        rw [List.range_succ_eq_map, List.map_cons, List.map_map]
        simp only [Nat.cast_zero, add_zero]
        congr 1
        apply List.map_congr_left
        intro offset _
        dsimp
        congr 1
        ring
      rw [mapped]
      by_cases done : 10000 ≤ polygonal s n
      · have rejected (offset : Nat) : ¬ accepted (polygonal s (n + offset)) := by
          have above := done.trans (monotone_from s n sides positive offset)
          intro accepted
          have upper := accepted.2.1
          omega
        have first : ¬ accepted (polygonal s n) := by simpa using rejected 0
        have tail :
            ((List.range fuel).map (fun offset : Nat => polygonal s (n + 1 + offset))).filter
              (fun value => decide (accepted value)) = [] := by
          apply List.filter_eq_nil_iff.mpr
          intro value member
          obtain ⟨offset, _, rfl⟩ := List.mem_map.mp member
          have absent : ¬ accepted (polygonal s (n + 1 + offset)) := by
            simpa [Nat.cast_add, Nat.cast_one, add_assoc, add_comm, add_left_comm] using rejected (offset + 1)
          simpa using absent
        simp [collect, done, first, tail]
      · rw [collect, if_neg done, ih (n + 1) (by omega)]
        by_cases keep : accepted (polygonal s n) <;> simp [emitted, keep]

/-- The generated list contains every qualifying positive-index polygonal number. -/
theorem mem_answer (s value : Int) (sides : 3 ≤ s) :
    value ∈ answer s ↔
      ∃ n : Nat, 1 ≤ n ∧ n ≤ 140 ∧ polygonal s n = value ∧ accepted value := by
  rw [answer, collect_eq_window 140 s 1 sides (by decide)]
  simp only [List.mem_filter, decide_eq_true_eq, List.mem_map, List.mem_range]
  constructor
  · rintro ⟨⟨offset, bound, equal⟩, keep⟩
    refine ⟨offset + 1, by omega, by omega, ?_, keep⟩
    simpa [Nat.cast_add, Nat.cast_one, add_comm] using equal
  · rintro ⟨n, positive, upper, equal, keep⟩
    refine ⟨⟨n - 1, by omega, ?_⟩, keep⟩
    have index : (1 : Int) + ↑(n - 1) = n := by omega
    simpa [index] using equal

/-- No positive index outside the finite window can add a qualifying value. -/
theorem mem_answer_iff (s value : Int) (sides : 3 ≤ s) :
    value ∈ answer s ↔
      ∃ n : Nat, 1 ≤ n ∧ polygonal s n = value ∧ accepted value := by
  rw [mem_answer s value sides]
  constructor
  · rintro ⟨n, positive, _, equal, keep⟩
    exact ⟨n, positive, equal, keep⟩
  · rintro ⟨n, positive, equal, keep⟩
    refine ⟨n, positive, ?_, equal, keep⟩
    by_contra outside
    have above := data.«project-euler-statements».data.py.project_euler_statements.p61.polygonal.above_four_digits
      s n sides (by omega)
    have upper := keep.2.1
    change 10000 ≤ polygonal s n at above
    rw [equal] at above
    omega

private def numbers (values : List Int) : List Mutable.Value := values.map number
private def working (base : Mutable.Heap) (values : List Int) : Mutable.Heap :=
  base ++ [.list (numbers values)]

private def rangeCondition : Expr :=
  .compare (.constant (.int 1000)) [(.ltE, .name "val"), (.ltE, .constant (.int 9999))]
private def suffixCondition : Expr :=
  .compare (.binOp (.name "val") .mod (.constant (.int 100))) [(.gtE, .constant (.int 10))]
private def filterCode : List Stmt :=
  [.ifThen rangeCondition
    [.ifThen suffixCondition [.expr (.call (.attribute (.name "nums") "append") [.name "val"])] []] []]
private def body : List Stmt :=
  [.assign [.name "val"] (.call (.name "polygonal") [.name "s", .name "n"]),
   .ifThen (.compare (.name "val") [(.gtE, .constant (.int 10000))]) [.breakLoop] []] ++
  filterCode ++ [.augAssign (.name "n") .add (.constant (.int 1))]
private def statement : Stmt := .whileDo (.constant (.bool true)) body []
private def code : List Stmt :=
  [.annAssign (.name "nums") (.subscript (.name "list") (.name "int")) (some (.list [])),
   .assign [.name "n"] (.constant (.int 1)), statement, .return (.name "nums")]

theorem code_source :
    executableBody ((findFunctionIn program.body "generate_4digit_polygonals").getD ([], [])).2 = code := rfl

private theorem range_evaluates (value : Int) (env : HeapCalls.Env) (heap : Mutable.Heap)
    (getVal : env "val" = some (number value)) (globals : Mutable.Globals) :
    Evaluates program env rangeCondition heap (flag (decide (1000 ≤ value ∧ value ≤ 9999))) heap globals := by
  by_cases lower : 1000 ≤ value
  · simp only [lower, true_and]
    apply Evaluates.chain (middle := heap) Evaluates.constant
    apply Evaluates.Compares.next (middle := heap) (afterComparison := heap) (compared := flag true)
      (Evaluates.lookup getVal)
    · simp [HeapCalls.compare, flag, scalarCompare, evalCmpOp, lower]
    · rfl
    · apply Evaluates.Compares.last (middle := heap) (finalHeap := heap) Evaluates.constant
      simp [HeapCalls.compare, flag, scalarCompare, evalCmpOp]
  · simp only [lower, false_and, decide_false]
    apply Evaluates.chain (middle := heap) Evaluates.constant
    apply Evaluates.Compares.stop (middle := heap) (finalHeap := heap) (Evaluates.lookup getVal)
    · simp [HeapCalls.compare, flag, scalarCompare, evalCmpOp, lower]
    · rfl

private theorem suffix_evaluates (value : Int) (env : HeapCalls.Env) (heap : Mutable.Heap)
    (getVal : env "val" = some (number value)) (globals : Mutable.Globals) :
    Evaluates program env suffixCondition heap (flag (decide (10 ≤ value % 100))) heap globals := by
  apply Evaluates.compare (middle := heap) (afterRight := heap) (finalHeap := heap)
    (av := number (value % 100)) (bv := number 10)
  · apply Evaluates.binary (middle := heap) (afterRight := heap) (finalHeap := heap)
      (Evaluates.lookup getVal) Evaluates.constant
    simp [Mutable.binary, scalarBinary, number]
  · exact Evaluates.constant
  · simp [HeapCalls.compare, flag, scalarCompare, evalCmpOp]

/-- The source's two filters append exactly one qualifying value, without allocating. -/
private theorem filter_step (value : Int) (env : HeapCalls.Env) (base : Mutable.Heap)
    (before : List Int) (getVal : env "val" = some (number value))
    (getNums : env "nums" = some (.ref base.length)) (globals : Mutable.Globals) :
    BlockContinues program env filterCode (working base before) env
      (working base (before ++ emitted value)) globals := by
  have rangeRun := range_evaluates value env (working base before) getVal globals
  have suffixRun := suffix_evaluates value env (working base before) getVal globals
  by_cases range : 1000 ≤ value ∧ value ≤ 9999
  · by_cases suffix : 10 ≤ value % 100
    · simp only [emitted, accepted, range.1, range.2, suffix, and_self, ↓reduceIte]
      apply BlockContinues.branch (flag := true) (by simpa [range] using rangeRun) rfl
      apply BlockContinues.branch (flag := true) (by simpa [suffix] using suffixRun) rfl
      have found : (working base before)[base.length]? = some (.list (numbers before)) := by simp [working]
      simpa [working, numbers, List.set_append_right] using
        BlockContinues.discard (Evaluates.append (Evaluates.lookup getNums) found (Evaluates.lookup getVal) found)
    · have rejected : ¬ accepted value := by simp [accepted, suffix]
      simp only [emitted, if_neg rejected, List.append_nil]
      apply BlockContinues.branch (flag := true) (by simpa [range] using rangeRun) rfl
      apply BlockContinues.branch (flag := false) (by simpa [suffix] using suffixRun) rfl
      exact .nil
  · have rejected : ¬ accepted value := fun h => range ⟨h.1, h.2.1⟩
    simp only [emitted, if_neg rejected, List.append_nil]
    apply BlockContinues.branch (flag := false) (by simpa [range] using rangeRun) rfl
    exact .nil

private theorem polygonal_call (s n : Int) (env : HeapCalls.Env) (heap : Mutable.Heap)
    (getS : env "s" = some (number s)) (getN : env "n" = some (number n))
    (globals : Mutable.Globals) :
    Evaluates program env (.call (.name "polygonal") [.name "s", .name "n"]) heap
      (number (polygonal s n)) heap globals :=
  Evaluates.call (.cons (Evaluates.lookup getS) (.cons (Evaluates.lookup getN) .nil))
    (solvers.«61».polygonal.heap_result s n heap globals)

private theorem loop_done (s n : Int) (done : 10000 ≤ polygonal s n)
    (env : HeapCalls.Env) (base : Mutable.Heap) (before : List Int)
    (getS : env "s" = some (number s)) (getN : env "n" = some (number n))
    (getNums : env "nums" = some (.ref base.length)) (globals : Mutable.Globals) :
    BlockReturns program env [statement, .return (.name "nums")] (working base before)
      (.ref base.length) (working base before) globals := by
  apply BlockReturns.while_loop (flag := true) (middle := working base before) Evaluates.constant rfl
  apply BlockReturns.assign (polygonal_call s n env (working base before) getS getN globals)
  apply BlockReturns.branch (flag := true) (value := flag true) (middle := working base before)
    (Evaluates.compare (middle := working base before) (afterRight := working base before)
      (av := number (polygonal s n)) (bv := number 10000)
      (Evaluates.lookup rfl) Evaluates.constant
      (by simp [HeapCalls.compare, scalarCompare, evalCmpOp, flag, done])) rfl
  apply BlockReturns.break_loop (next := statement) (following := [.return (.name "nums")]) rfl
  exact BlockReturns.return_value (Evaluates.lookup (by simpa [setLocal] using getNums))

/-- Every actual loop call reaches its break by index 141 and preserves the caller heap prefix. -/
theorem loop (fuel : Nat) (s n : Int) (sides : 3 ≤ s) (positive : 1 ≤ n)
    (bound : 141 ≤ n + fuel) (env : HeapCalls.Env) (base : Mutable.Heap) (before : List Int)
    (getS : env "s" = some (number s)) (getN : env "n" = some (number n))
    (getNums : env "nums" = some (.ref base.length)) (globals : Mutable.Globals := []) :
    BlockReturns program env [statement, .return (.name "nums")] (working base before)
      (.ref base.length) (working base (before ++ collect fuel s n)) globals := by
  induction fuel generalizing n env before with
  | zero =>
      have done := data.«project-euler-statements».data.py.project_euler_statements.p61.polygonal.above_four_digits
        s n sides (by simpa using bound)
      simpa [collect] using loop_done s n done env base before getS getN getNums globals
  | succ fuel ih =>
      by_cases done : 10000 ≤ polygonal s n
      · simpa [collect, done] using loop_done s n done env base before getS getN getNums globals
      · let valued := setLocal env "val" (number (polygonal s n))
        let nextEnv := setLocal valued "n" (number (n + 1))
        have valuedS : valued "s" = some (number s) := by simpa [valued, setLocal] using getS
        have valuedN : valued "n" = some (number n) := by simpa [valued, setLocal] using getN
        have valuedNums : valued "nums" = some (.ref base.length) := by simpa [valued, setLocal] using getNums
        have nextS : nextEnv "s" = some (number s) := by simpa [nextEnv, setLocal] using valuedS
        have nextN : nextEnv "n" = some (number (n + 1)) := by simp [nextEnv, setLocal]
        have nextNums : nextEnv "nums" = some (.ref base.length) := by simpa [nextEnv, setLocal] using valuedNums
        have filtered := filter_step (polygonal s n) valued base before rfl valuedNums globals
        have rest := ih (n + 1) (by omega) (by omega) nextEnv
          (before ++ emitted (polygonal s n)) nextS nextN nextNums
        simp only [collect, if_neg done]
        apply BlockReturns.while_loop (flag := true) (middle := working base before) Evaluates.constant rfl
        apply BlockReturns.assign (polygonal_call s n env (working base before) getS getN globals)
        apply BlockReturns.branch (flag := false) (value := flag false) (middle := working base before)
          (Evaluates.compare (middle := working base before) (afterRight := working base before)
            (av := number (polygonal s n)) (bv := number 10000)
            (Evaluates.lookup rfl) Evaluates.constant
            (by simp [HeapCalls.compare, scalarCompare, evalCmpOp, flag, done])) rfl
        apply filtered.returns
        apply BlockReturns.augAssign valuedN Evaluates.constant
          (value := number (n + 1)) (afterOperation := working base (before ++ emitted (polygonal s n)))
        · simp [HeapCalls.inplace, Mutable.binary, scalarBinary, number]
        · apply BlockReturns.loop_resume
          simpa [nextEnv, List.append_assoc] using rest

/-- The unchanged Python helper returns its exact list for every valid polygonal family. -/
theorem result (s : Int) (sides : 3 ≤ s) (heap : Mutable.Heap) (globals : Mutable.Globals := []) :
    ReturnsValue program "generate_4digit_polygonals" [number s] heap (.ref heap.length)
      (working heap (answer s)) globals := by
  let params := ((findFunctionIn program.body "generate_4digit_polygonals").getD ([], [])).1
  let sourceBody := ((findFunctionIn program.body "generate_4digit_polygonals").getD ([], [])).2
  let env := functionEnv params sourceBody [number s] globals
  apply BlockReturns.function rfl rfl
  change BlockReturns program env (executableBody sourceBody) heap (.ref heap.length) (working heap (answer s)) globals
  rw [code_source]
  have allocated : BlockContinues program env
      [.annAssign (.name "nums") (.subscript (.name "list") (.name "int")) (some (.list []))]
      heap (setLocal env "nums" (.ref heap.length)) (working heap []) globals := by
    exact BlockContinues.annAssign (Evaluates.list .nil)
  apply allocated.returns
  apply BlockReturns.assign Evaluates.constant
  exact loop 140 s 1 sides (by decide) (by decide)
    (setLocal (setLocal env "nums" (.ref heap.length)) "n" (number 1)) heap []
    (by cases globals <;> rfl) rfl rfl globals

end solvers.«61».generate_4digit_polygonals

namespace solvers.«61»

theorem generate_4digit_polygonals_source_result (s : Int) (heap : FormalSW.PyAst.Mutable.Heap)
    (globals : FormalSW.PyAst.Mutable.Globals) :
    3 ≤ s → FormalSW.PyAst.HeapCalls.ReturnsValue Solvers.P61Ast.module "generate_4digit_polygonals"
      [.scalar (.int s)] heap (.ref heap.length)
      (heap ++ [.list ((solvers.«61».generate_4digit_polygonals.answer s).map
        (fun value => .scalar (.int value)))]) globals := by
  intro sides
  exact solvers.«61».generate_4digit_polygonals.result s sides heap globals

end solvers.«61»

namespace solvers.«61».build_prefix_map
open FormalSW.PyAst FormalSW.PyAst.HeapCalls

private abbrev program := Solvers.P61Ast.module
private abbrev number (value : Int) : Mutable.Value := .scalar (.int value)
private def numbers (values : List Int) : List Mutable.Value := values.map number

def bucket (values : List Int) (key : Int) : List Int :=
  values.filter (fun value => decide (value / 100 = key))

def lookup : List (Int × Nat) → Int → Option Nat
  | [], _ => none
  | (stored, address) :: rest, key => if stored = key then some address else lookup rest key

private def encode (entries : List (Int × Nat)) : List (Constant × Mutable.Value) :=
  entries.map (fun (key, address) => (.int key, .ref address))

private theorem lookup_encode (entries : List (Int × Nat)) (key : Int) :
    ScalarDict.lookup (encode entries) (.int key) = (lookup entries key).map Mutable.Value.ref := by
  induction entries with
  | nil => rfl
  | cons entry rest ih =>
      obtain ⟨stored, address⟩ := entry
      by_cases same : stored = key
      · simp [encode, lookup, ScalarDict.lookup, scalarCompare, evalCmpOp, same]
      · simpa [encode, lookup, ScalarDict.lookup, scalarCompare, evalCmpOp, same] using ih

private theorem lookup_append (left right : List (Int × Nat)) (key : Int) :
    lookup (left ++ right) key =
      match lookup left key with
      | some address => some address
      | none => lookup right key := by
  induction left with
  | nil => rfl
  | cons entry rest ih =>
      obtain ⟨stored, address⟩ := entry
      by_cases same : stored = key <;> simp [lookup, same, ih]

private theorem insert_encode (entries : List (Int × Nat)) (key : Int) (address : Nat)
    (absent : lookup entries key = none) :
    ScalarDict.insert (encode entries) (.int key) (.ref address) = encode (entries ++ [(key, address)]) := by
  induction entries with
  | nil => rfl
  | cons entry rest ih =>
      obtain ⟨stored, previous⟩ := entry
      by_cases same : stored = key
      · simp [lookup, same] at absent
      · have tail : lookup rest key = none := by simpa [lookup, same] using absent
        simpa [encode, ScalarDict.insert, scalarCompare, evalCmpOp, same] using
          congrArg (fun result : List (Constant × Mutable.Value) => (.int stored, .ref previous) :: result) (ih tail)

private theorem found_append (heap extra : Mutable.Heap) (address : Nat) (object : Mutable.Object)
    (found : heap[address]? = some object) : (heap ++ extra)[address]? = some object := by
  rw [List.getElem?_append_left (List.getElem?_eq_some_iff.mp found).1]
  exact found

private theorem extension_set (base heap : Mutable.Heap)
    (extension : ∃ extra, heap = base ++ extra) (address : Nat) (outside : base.length ≤ address)
    (object : Mutable.Object) : ∃ extra, heap.set address object = base ++ extra := by
  obtain ⟨extra, rfl⟩ := extension
  exact ⟨extra.set (address - base.length) object, List.set_append_right address object outside⟩

/-- The source table owns distinct fresh list objects. Its buckets retain input order. -/
structure Store (base heap : Mutable.Heap) (processed : List Int) where
  entries : List (Int × Nat)
  extension : ∃ extra, heap = base ++ extra
  table : heap[base.length]? = some (.dict (encode entries))
  present : ∀ key address, lookup entries key = some address →
    base.length < address ∧ heap[address]? = some (.list (numbers (bucket processed key)))
  exclusive : ∀ left right address,
    lookup entries left = some address → lookup entries right = some address → left = right
  missing : ∀ key, lookup entries key = none → bucket processed key = []

def Store.initial (base : Mutable.Heap) : Store base (base ++ [.dict []]) [] where
  entries := []
  extension := ⟨[.dict []], rfl⟩
  table := by simp [encode]
  present := by intro key address found; simp [lookup] at found
  exclusive := by intro left right address found; simp [lookup] at found
  missing := by intro key _; rfl

variable {base heap : Mutable.Heap} {processed : List Int}

def Store.append (store : Store base heap processed) (extra : Mutable.Heap) :
    Store base (heap ++ extra) processed where
  entries := store.entries
  extension := by
    obtain ⟨old, rfl⟩ := store.extension
    exact ⟨old ++ extra, List.append_assoc _ _ _⟩
  table := found_append heap extra _ _ store.table
  present := by
    intro key address looked
    obtain ⟨outside, found⟩ := store.present key address looked
    exact ⟨outside, found_append heap extra address _ found⟩
  exclusive := store.exclusive
  missing := store.missing

/-- Eager evaluation allocates a default even if another bucket already exists. -/
def Store.add_empty (store : Store base heap processed) (key : Int) (absent : lookup store.entries key = none) :
    Store base ((heap ++ [Mutable.Object.list []]).set base.length
      (.dict (encode (store.entries ++ [(key, heap.length)])))) processed := by
  let entries : List (Int × Nat) := store.entries ++ [(key, heap.length)]
  let extended : Mutable.Heap := heap ++ [.list []]
  let updated : Mutable.Heap := extended.set base.length (.dict (encode entries))
  have tableInside : base.length < heap.length := (List.getElem?_eq_some_iff.mp store.table).1
  have classify (query : Int) (address : Nat) (looked : lookup entries query = some address) :
      lookup store.entries query = some address ∨ (query = key ∧ address = heap.length) := by
    cases old : lookup store.entries query with
    | some previous =>
        have same : previous = address := by simpa [entries, lookup_append, old] using looked
        exact Or.inl (by simp [same])
    | none =>
        by_cases same : key = query
        · subst query
          have fresh : heap.length = address := by simpa [entries, lookup_append, old, lookup] using looked
          exact Or.inr ⟨rfl, fresh.symm⟩
        · simp [entries, lookup_append, old, lookup, same] at looked
  refine { entries := entries, extension := ?_, table := ?_, present := ?_, exclusive := ?_, missing := ?_ }
  · exact extension_set base extended (store.append [.list []]).extension _ (Nat.le_refl _) _
  · exact List.getElem?_set_self (by simp only [List.length_append, List.length_singleton]; omega)
  · intro query address looked
    rcases classify query address looked with old | ⟨rfl, rfl⟩
    · obtain ⟨outside, found⟩ := store.present query address old
      refine ⟨outside, ?_⟩
      simpa only [updated, List.getElem?_set_ne (by omega : base.length ≠ address)] using
        found_append heap [.list []] address _ found
    · refine ⟨tableInside, ?_⟩
      have empty := store.missing _ absent
      rw [List.getElem?_set_ne (by omega : base.length ≠ heap.length)]
      simp [empty, numbers]
  · intro left right address leftFound rightFound
    rcases classify left address leftFound with oldLeft | ⟨rfl, freshLeft⟩
    · rcases classify right address rightFound with oldRight | ⟨rfl, freshRight⟩
      · exact store.exclusive left right address oldLeft oldRight
      · have found := (store.present left address oldLeft).2
        have inside := (List.getElem?_eq_some_iff.mp found).1
        omega
    · rcases classify right address rightFound with oldRight | ⟨rfl, _⟩
      · have found := (store.present right address oldRight).2
        have inside := (List.getElem?_eq_some_iff.mp found).1
        omega
      · rfl
  · intro query missing
    have old : lookup store.entries query = none := by
      cases looked : lookup store.entries query with
      | none => rfl
      | some address => simp [entries, lookup_append, looked] at missing
    exact store.missing query old

/-- Mutating one bucket preserves the table, all other buckets, and the caller prefix. -/
def Store.bump (store : Store base heap processed) (value : Int) (address : Nat)
    (looked : lookup store.entries (value / 100) = some address) :
    Store base (heap.set address (.list (numbers (bucket processed (value / 100) ++ [value]))))
      (processed ++ [value]) := by
  obtain ⟨outside, found⟩ := store.present (value / 100) address looked
  refine { entries := store.entries
           extension := ?_
           table := ?_
           present := ?_
           exclusive := store.exclusive
           missing := ?_ }
  · exact extension_set base heap store.extension address (by omega) _
  · simpa only [List.getElem?_set_ne (by omega : address ≠ base.length)] using store.table
  · intro key other otherLooked
    obtain ⟨otherOutside, otherFound⟩ := store.present key other otherLooked
    refine ⟨otherOutside, ?_⟩
    by_cases same : key = value / 100
    · subst key
      have equal : other = address := Option.some.inj (otherLooked.symm.trans looked)
      subst other
      simpa [bucket] using List.getElem?_set_self (a := .list (numbers (bucket processed (value / 100) ++ [value])))
        (List.getElem?_eq_some_iff.mp found).1
    · have separate : address ≠ other := by
        intro equal
        subst other
        exact same (store.exclusive key (value / 100) address otherLooked looked)
      have different : value / 100 ≠ key := Ne.symm same
      simpa [bucket, different, List.getElem?_set_ne separate] using otherFound
  · intro key absent
    have different : value / 100 ≠ key := by
      intro equal
      subst key
      rw [looked] at absent
      cases absent
    simpa [bucket, different] using store.missing key absent

private def defaultExpression : Expr :=
  .call (.attribute (.name "mp") "setdefault") [.name "pref", .list []]

private theorem default_evaluates (store : Store base heap processed) (key : Int) (env : HeapCalls.Env)
    (getMap : env "mp" = some (.ref base.length)) (getKey : env "pref" = some (number key))
    (globals : Mutable.Globals) :
    ∃ (middle : Mutable.Heap) (address : Nat) (ready : Store base middle processed),
      lookup ready.entries key = some address ∧
      Evaluates program env defaultExpression heap (.ref address) middle globals := by
  have table := found_append heap [.list []] base.length _ store.table
  cases old : lookup store.entries key with
  | none =>
      let middle : Mutable.Heap := (heap ++ [Mutable.Object.list []]).set base.length
        (.dict (encode (store.entries ++ [(key, heap.length)])))
      let ready := store.add_empty key old
      refine ⟨middle, heap.length, ready, ?_, ?_⟩
      · change lookup (store.entries ++ [(key, heap.length)]) key = some heap.length
        simp [lookup_append, old, lookup]
      · apply Evaluates.dictionary_setdefault (address := base.length) (afterReceiver := heap)
          (afterArguments := heap ++ [.list []]) (values := [number key, .ref heap.length])
          (by intro name same; cases same; rw [getMap]; simp) (Evaluates.lookup getMap)
        · simp [bindMethod, store.table]
        · exact .cons (Evaluates.lookup getKey) (.cons (Evaluates.list .nil) .nil)
        · change Mutable.dictionarySetDefault base.length (number key) (.ref heap.length)
            (heap ++ [.list []]) = _
          rw [Mutable.dictionarySetDefault_scalar _ _ _ table (.int key) (.ref heap.length), lookup_encode, old]
          simp [middle, insert_encode _ _ _ old]
  | some address =>
      let ready := store.append [.list []]
      refine ⟨heap ++ [.list []], address, ready, old, ?_⟩
      apply Evaluates.dictionary_setdefault (address := base.length) (afterReceiver := heap)
        (afterArguments := heap ++ [.list []]) (values := [number key, .ref heap.length])
        (by intro name same; cases same; rw [getMap]; simp) (Evaluates.lookup getMap)
      · simp [bindMethod, store.table]
      · exact .cons (Evaluates.lookup getKey) (.cons (Evaluates.list .nil) .nil)
      · change Mutable.dictionarySetDefault base.length (number key) (.ref heap.length)
          (heap ++ [.list []]) = _
        rw [Mutable.dictionarySetDefault_scalar _ _ _ table (.int key) (.ref heap.length), lookup_encode, old]
        rfl

private def appendCode : List Stmt :=
  [.expr (.call (.attribute defaultExpression "append") [.name "x"])]

private theorem emit (store : Store base heap processed) (value : Int) (env : HeapCalls.Env)
    (getMap : env "mp" = some (.ref base.length)) (getKey : env "pref" = some (number (value / 100)))
    (getValue : env "x" = some (number value)) (globals : Mutable.Globals) :
    ∃ (finalHeap : Mutable.Heap) (_finalStore : Store base finalHeap (processed ++ [value])),
      BlockContinues program env appendCode heap env finalHeap globals := by
  obtain ⟨middle, address, ready, looked, prepared⟩ := default_evaluates store (value / 100) env getMap getKey globals
  have found := (ready.present (value / 100) address looked).2
  refine ⟨_, ready.bump value address looked, ?_⟩
  simpa only [numbers, List.map_append, List.map_cons, List.map_nil] using
    BlockContinues.discard (Evaluates.append prepared found (Evaluates.lookup getValue) found)

private def body : List Stmt :=
  [.assign [.name "pref"] (.binOp (.name "x") .floorDiv (.constant (.int 100)))] ++ appendCode
private def annotation : Expr :=
  .subscript (.name "dict") (.tuple [.name "int", .subscript (.name "list") (.name "int")])
private def code : List Stmt :=
  [.annAssign (.name "mp") annotation (some (.dict [])),
   .forIn (.name "x") (.name "nums") body [], .return (.name "mp")]

theorem code_source :
    executableBody ((findFunctionIn program.body "build_prefix_map").getD ([], [])).2 = code := rfl

private theorem source_found (store : Store base heap processed) (address : Nat) (values : List Int)
    (found : base[address]? = some (.list (numbers values))) :
    heap[address]? = some (.list (numbers values)) := by
  obtain ⟨extra, rfl⟩ := store.extension
  exact found_append base extra address _ found

/-- The actual source walks the original list and appends each number to its prefix bucket. -/
theorem loop (remaining values : List Int) (position : Nat) (suffix : values.drop position = remaining)
    (store : Store base heap processed) (input : Nat) (found : base[input]? = some (.list (numbers values)))
    (env : HeapCalls.Env) (getMap : env "mp" = some (.ref base.length)) (globals : Mutable.Globals := []) :
    ∃ (finalHeap : Mutable.Heap) (_finalStore : Store base finalHeap (processed ++ remaining)),
      BlockReturns program env
        [.sequenceStepLoop "x" (.ref input) position body [], .return (.name "mp")]
        heap (.ref base.length) finalHeap globals := by
  induction remaining generalizing position heap processed env with
  | nil =>
      refine ⟨heap, ?_, ?_⟩
      · simpa only [List.append_nil] using store
      · apply BlockReturns.sequence_done (items := numbers values)
          (by simp [sequenceValue, Mutable.sequence, source_found store input values found])
        · have done : values[position]? = none := by rw [← List.head?_drop, suffix]; rfl
          simp [numbers, List.getElem?_map, done]
        · exact BlockReturns.return_value (Evaluates.lookup getMap)
  | cons value rest ih =>
      let bound := setLocal env "x" (number value)
      let prefixed := setLocal bound "pref" (number (value / 100))
      have boundMap : bound "mp" = some (.ref base.length) := by simpa [bound, setLocal] using getMap
      have prefixedMap : prefixed "mp" = some (.ref base.length) := by simpa [prefixed, setLocal] using boundMap
      have prefixRun : BlockContinues program bound
          [.assign [.name "pref"] (.binOp (.name "x") .floorDiv (.constant (.int 100)))] heap prefixed heap globals := by
        apply BlockContinues.assign
        apply Evaluates.binary (middle := heap) (afterRight := heap) (finalHeap := heap)
          (av := number value) (bv := number 100) (Evaluates.lookup rfl) Evaluates.constant
        simp [Mutable.binary, scalarBinary, number]
      obtain ⟨middle, ready, emitted⟩ := emit store value prefixed prefixedMap rfl
        (by simp [prefixed, bound, setLocal]) globals
      obtain ⟨finalHeap, finishedStore, executed⟩ := ih (position + 1)
        (by simp only [List.drop_add_one_eq_tail_drop, suffix, List.tail_cons]) ready prefixed prefixedMap
      refine ⟨finalHeap, ?_, ?_⟩
      · simpa only [List.append_assoc, List.singleton_append] using finishedStore
      · apply BlockReturns.sequence_step (items := numbers values) (value := number value)
          (by simp [sequenceValue, Mutable.sequence, source_found store input values found])
        · have item : values[position]? = some value := by rw [← List.head?_drop, suffix]; rfl
          simp [numbers, List.getElem?_map, item]
        · simpa [body, Calls.loopContinuation, Calls.loopControl, Calls.loopControlStmt, appendCode,
            List.append_assoc] using (prefixRun.append emitted).returns executed

/-- Arbitrary integer inputs produce exactly their prefix buckets in fresh heap objects. -/
theorem result (values : List Int) (heap : Mutable.Heap) (input : Nat)
    (found : heap[input]? = some (.list (numbers values))) (globals : Mutable.Globals := []) :
    ∃ (finalHeap : Mutable.Heap) (_finalStore : Store heap finalHeap values),
      ReturnsValue program "build_prefix_map" [.ref input] heap (.ref heap.length) finalHeap globals := by
  let params := ((findFunctionIn program.body "build_prefix_map").getD ([], [])).1
  let sourceBody := ((findFunctionIn program.body "build_prefix_map").getD ([], [])).2
  let env := functionEnv params sourceBody [.ref input] globals
  let initialized := setLocal env "mp" (.ref heap.length)
  let initial := Store.initial heap
  obtain ⟨finalHeap, finalStore, executed⟩ := loop values values 0 rfl initial input found initialized rfl globals
  refine ⟨finalHeap, by simpa using finalStore, ?_⟩
  apply BlockReturns.function rfl rfl
  change BlockReturns program env (executableBody sourceBody) heap (.ref heap.length) finalHeap globals
  rw [code_source]
  have allocated : BlockContinues program env [.annAssign (.name "mp") annotation (some (.dict []))]
      heap initialized (heap ++ [.dict []]) globals := BlockContinues.annAssign (Evaluates.dictionary .nil)
  apply allocated.returns
  apply BlockReturns.for_sequence (items := numbers values) (by intros; intro equal; cases equal)
    (Evaluates.lookup (by cases globals <;> rfl))
    (by simp [Mutable.sequence, source_found initial input values found])
  exact executed

end solvers.«61».build_prefix_map

namespace solvers.«61»

theorem build_prefix_map_source_result (values : List Int) (heap : FormalSW.PyAst.Mutable.Heap) (input : Nat)
    (globals : FormalSW.PyAst.Mutable.Globals) :
    heap[input]? = some (.list (values.map (fun value => .scalar (.int value)))) →
    ∃ finalHeap, Nonempty (build_prefix_map.Store heap finalHeap values) ∧
      FormalSW.PyAst.HeapCalls.ReturnsValue Solvers.P61Ast.module "build_prefix_map"
        [.ref input] heap (.ref heap.length) finalHeap globals := by
  intro found
  obtain ⟨finalHeap, store, returned⟩ := build_prefix_map.result values heap input found globals
  exact ⟨finalHeap, ⟨store⟩, returned⟩

end solvers.«61»
