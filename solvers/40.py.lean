import «.formalsw».solvers.«40.py.ast»
import lang.py.FormalSW.HeapCalls
import lang.py.FormalSW.IntegerParsing
import lang.py.FormalSW.TruthCalls
import lang.py.FormalSW.ContainerTypes
import ProjectEulerStatements.Lib.ChampernowneBlocks
import ProjectEulerStatements.Lib.ChampernowneStrings
import data.«project-euler-statements».data.py.project_euler_statements.«p40.py»

set_option Elab.async false

namespace solvers.«40»
open FormalSW.PyAst FormalSW.PyAst.HeapCalls
open ProjectEulerStatements ProjectEulerStatements.ChampernowneBlocks
private abbrev program := Solvers.P40Ast.module
private abbrev number (n : Nat) : Mutable.Value := .scalar (.int n)

namespace digit_at
private abbrev code : List Stmt :=
  executableBody ((findFunctionIn program.body "digit_at").getD ([], [])).2
private abbrev body : List Stmt :=
  match (code[4]? : Option Stmt) with | some (.whileDo _ statements _) => statements | _ => []
private abbrev finish : List Stmt :=
  match (body[1]? : Option Stmt) with | some (.ifThen _ _ statements) => statements | _ => []
private abbrev skip : List Stmt :=
  match (body[1]? : Option Stmt) with | some (.ifThen _ statements _) => statements | _ => []
private def Locals (env : HeapCalls.Env) (width rank : Nat) : Prop :=
  env "n" = some (number rank) ∧ env "length" = some (number width) ∧
    env "start" = some (number (blockStart width)) ∧
    env "count" = some (number (blockCount width))

private theorem indexed_character (n index : Nat) (positive : 0 < n)
    (bounded : index < (P40.digitsBE n).length) (heap : Mutable.Heap) :
    Mutable.getItem (.scalar (.string (toString (n : Int)))) (number index) heap =
      some (.ok (.scalar (.string (toString ((P40.digitsBE n).getD index 0)))), heap) := by
  have lengthEqual : (toString (n : Int)).toList.length = (P40.digitsBE n).length := by
    rw [ChampernowneStrings.int_characters n positive, List.length_map]
  have nonnegative : ¬ (index : Int) < 0 := by omega
  have normalized : FormalSW.Py.Builtin.Sequence.normalizeIndex
      (toString (n : Int)).toList.length (index : Int) = some index := by
    have inside : (index : Int) < ((toString (n : Int)).toList.length : Int) := by omega
    simp only [FormalSW.Py.Builtin.Sequence.normalizeIndex, nonnegative, not_le.mpr inside,
      decide_false, Bool.false_or, Bool.false_eq_true, if_false, Int.toNat_natCast]
  have fetched : FormalSW.Py.Builtin.Sequence.get? (toString (n : Int)).toList (index : Int) =
      some (Nat.digitChar ((P40.digitsBE n).getD index 0)) := by
    simpa only [FormalSW.Py.Builtin.Sequence.get?, normalized, Option.bind_some] using
      ChampernowneStrings.int_character n index positive bounded
  have got : Strings.get (.string (toString (n : Int))) (.int index) =
      .ok (.string (toString ((P40.digitsBE n).getD index 0))) := by
    simp only [Strings.get, scalarNumber, fetched]
    rw [ChampernowneStrings.indexed_digit_string n index bounded]
  simp only [Mutable.getItem, got]
  rfl

private theorem digit_expression (n index : Nat) (positive : 0 < n)
    (bounded : index < (P40.digitsBE n).length) (heap : Mutable.Heap) (env : HeapCalls.Env)
    (getNum : env "num" = some (number n)) (getPos : env "pos" = some (number index)) :
    Evaluates program env (.builtinCall .int
      [.subscript (.builtinCall .str [.name "num"]) (.name "pos")]) heap
      (number ((P40.digitsBE n).getD index 0)) heap := by
  apply Evaluates.builtin (.cons
    (Evaluates.subscript (by intros; simp)
      (Evaluates.builtin (.cons (Evaluates.lookup getNum) .nil) rfl)
      (Evaluates.lookup getPos) (indexed_character n index positive bounded heap)) .nil)
  simp only [primitive, List.mapM_cons, List.mapM_nil, pure_bind]
  rw [Calls.integerConversion_nat]
  rfl

private theorem finish_result (width rank : Nat) (positive : 0 < width) (rankPositive : 0 < rank)
    (within : rank ≤ blockSize width) (heap : Mutable.Heap) (env : HeapCalls.Env)
    (locals : Locals env width rank) (following : List Stmt) :
    BlockReturns program env (finish ++ following) heap
      (number ((P40.digitsBE (blockStart width + (rank-1)/width)).getD ((rank-1)%width) 0)) heap := by
  let index := (rank-1)/width
  let position := (rank-1)%width
  let value := blockStart width + index
  have indexBound : index < blockCount width := by
    apply (Nat.div_lt_iff_lt_mul positive).mpr
    have : rank ≤ blockCount width * width := within
    omega
  have valueLower : blockStart width ≤ value := Nat.le_add_right _ _
  have valuePositive : 0 < value := lt_of_lt_of_le (blockStart_positive width) valueLower
  have valueWidth : (P40.digitsBE value).length = width :=
    decimal_length width value positive valueLower (Nat.add_lt_add_left indexBound _)
  have positionBound : position < (P40.digitsBE value).length := by
    rw [valueWidth]
    exact Nat.mod_lt _ positive
  let e1 := setLocal env "idx" (number index)
  let e2 := setLocal e1 "pos" (number position)
  let e3 := setLocal e2 "num" (number value)
  have difference (e : HeapCalls.Env) (getN : e "n" = some (number rank)) :
      Evaluates program e (.binOp (.name "n") .sub (.constant (.int 1))) heap
        (number (rank-1)) heap :=
    Evaluates.binary (Evaluates.lookup getN) Evaluates.constant
      (by simp [Mutable.binary, scalarBinary, number, Nat.cast_sub (by omega : 1 ≤ rank)])
  have first : BlockContinues program env [finish[0]'(by decide)] heap e1 heap := by
    apply BlockContinues.assign
    exact Evaluates.binary (difference env locals.1) (Evaluates.lookup locals.2.1)
      (by simp [Mutable.binary, scalarBinary, number, index, positive, show width ≠ 0 by omega])
  have second : BlockContinues program e1 [finish[1]'(by decide)] heap e2 heap := by
    apply BlockContinues.assign
    exact Evaluates.binary (difference e1 (by simpa [e1, setLocal] using locals.1))
      (Evaluates.lookup (by simpa [e1, setLocal] using locals.2.1))
      (by simp [Mutable.binary, scalarBinary, number, position, positive, show width ≠ 0 by omega])
  have third : BlockContinues program e2 [finish[2]'(by decide)] heap e3 heap := by
    apply BlockContinues.assign
    exact Evaluates.binary (av := number (blockStart width)) (bv := number index)
      (Evaluates.lookup (by simpa [e2, e1, setLocal] using locals.2.2.1))
      (Evaluates.lookup (by simp [e2, e1, setLocal]))
      (by simp [Mutable.binary, scalarBinary, number, value])
  apply BlockContinues.returns (first.append (second.append third))
  exact BlockReturns.return_value
    (digit_expression value position valuePositive positionBound heap e3 rfl (by simp [e3, e2, setLocal]))

private theorem skip_result (width rank : Nat) (positive : 0 < width)
    (beyond : blockSize width < rank) (heap : Mutable.Heap) (env : HeapCalls.Env)
    (locals : Locals env width rank) (getBlock : env "block_digits" = some (number (blockSize width))) :
    ∃ finalEnv, BlockContinues program env skip heap finalEnv heap ∧
      Locals finalEnv (width+1) (rank-blockSize width) := by
  let e1 := setLocal env "n" (number (rank-blockSize width))
  let e2 := setLocal e1 "length" (number (width+1))
  let e3 := setLocal e2 "start" (number (blockStart (width+1)))
  let e4 := setLocal e3 "count" (number (blockCount (width+1)))
  have first : BlockContinues program env [skip[0]'(by decide)] heap e1 heap :=
    BlockContinues.augAssign locals.1 (Evaluates.lookup getBlock)
      (by simp [inplace, Mutable.binary, scalarBinary, number,
        Nat.cast_sub (by omega : blockSize width ≤ rank)])
  have second : BlockContinues program e1 [skip[1]'(by decide)] heap e2 heap :=
    BlockContinues.augAssign (by simpa [e1, setLocal] using locals.2.1) Evaluates.constant
      (by simp [inplace, Mutable.binary, scalarBinary, number])
  have third : BlockContinues program e2 [skip[2]'(by decide)] heap e3 heap :=
    BlockContinues.augAssign (by simpa [e2, e1, setLocal] using locals.2.2.1) Evaluates.constant
      (by simp [inplace, Mutable.binary, scalarBinary, number, next_start width positive])
  have fourth : BlockContinues program e3 [skip[3]'(by decide)] heap e4 heap :=
    BlockContinues.augAssign (by simpa [e3, e2, e1, setLocal] using locals.2.2.2) Evaluates.constant
      (by simp [inplace, Mutable.binary, scalarBinary, number, next_count width positive])
  refine ⟨e4, first.append (second.append (third.append fourth)), ?_⟩
  simp [Locals, e4, e3, e2, e1, setLocal]

private theorem loop (rank width : Nat) (rankPositive : 0 < rank) (positive : 0 < width)
    (heap : Mutable.Heap) (env : HeapCalls.Env) (locals : Locals env width rank) :
    BlockReturns program env [.whileDo (.constant (.bool true)) body []] heap
      (number (blockDigit width rank)) heap := by
  let bound := setLocal env "block_digits" (number (blockSize width))
  have assigned : BlockContinues program env [body[0]'(by decide)] heap bound heap := by
    apply BlockContinues.assign
    exact Evaluates.binary (Evaluates.lookup locals.2.2.2) (Evaluates.lookup locals.2.1)
      (by simp [Mutable.binary, scalarBinary, number, blockSize])
  have nextLocals : Locals bound width rank := by
    rcases locals with ⟨hn, hw, hs, hc⟩
    exact ⟨by simpa [bound, setLocal] using hn, by simpa [bound, setLocal] using hw,
      by simpa [bound, setLocal] using hs, by simpa [bound, setLocal] using hc⟩
  have tested : Evaluates program bound (.compare (.name "n") [(.gt, .name "block_digits")])
      heap (.scalar (.bool (decide (blockSize width < rank)))) heap :=
    Evaluates.compare (Evaluates.lookup nextLocals.1) (Evaluates.lookup rfl)
      (by simp [HeapCalls.compare, scalarCompare, evalCmpOp])
  rw [blockDigit_step width rank positive]
  apply BlockReturns.while_loop (flag := true) Evaluates.constant rfl
  apply BlockContinues.returns assigned
  by_cases beyond : blockSize width < rank
  · simp only [if_pos beyond]
    obtain ⟨nextEnv, advanced, nextState⟩ := skip_result width rank positive beyond heap bound nextLocals rfl
    have searched := loop (rank-blockSize width) (width+1) (by omega) (by omega) heap nextEnv nextState
    apply BlockReturns.branch (flag := true) (by simpa only [beyond, decide_true] using tested) rfl
    apply BlockContinues.returns advanced
    simpa only [Calls.loopContinuation, Calls.loopControl, Calls.loopControlStmt, List.nil_append,
      List.append_nil] using searched
  · simp only [if_neg beyond]
    apply BlockReturns.branch (flag := false) (by simpa only [beyond, decide_false] using tested) rfl
    exact finish_result width rank positive rankPositive (by omega) heap bound nextLocals _
termination_by rank
decreasing_by exact Nat.sub_lt rankPositive (blockSize_positive width positive)

/-- Execute the assertion and decimal block search without allocating a digit stream. -/
theorem result (n : Nat) (positive : 0 < n) (heap : Mutable.Heap) :
    ReturnsValue program "digit_at" [number n] heap (number (blockDigit 1 n)) heap := by
  let env := functionEnv ((findFunctionIn program.body "digit_at").getD ([], [])).1
    ((findFunctionIn program.body "digit_at").getD ([], [])).2 [number n] []
  let ready := setLocal (setLocal (setLocal env "length" (number 1)) "start" (number 1)) "count" (number 9)
  have getN : env "n" = some (number n) := rfl
  have initial : Locals ready 1 n := by
    exact ⟨by simpa [ready, setLocal] using getN, rfl, rfl, rfl⟩
  apply BlockReturns.function rfl rfl
  have asserted : Evaluates program env
      (.compare (.name "n") [(.gtE, .constant (.int 1))]) heap
      (.scalar (.bool (decide (1 ≤ n)))) heap :=
    Evaluates.compare (Evaluates.lookup getN) Evaluates.constant
      (by simp [HeapCalls.compare, scalarCompare, evalCmpOp])
  apply BlockReturns.assertion (value := .scalar (.bool true))
    (by simpa only [show 1 ≤ n by omega, decide_true] using asserted) rfl
  apply BlockReturns.assign Evaluates.constant
  apply BlockReturns.assign Evaluates.constant
  apply BlockReturns.assign Evaluates.constant
  exact loop n 1 positive (by decide) heap ready initial

end digit_at

private abbrev numbers (items : List Nat) : List Mutable.Value := items.map number
private abbrev answer := data.«project-euler-statements».data.py.project_euler_statements.p40.naive.answer

namespace solve
private def body : List Stmt := [.augAssign (.name "product") .mult (.call (.name "digit_at") [.name "position"])]

private theorem loop (items remaining : List Nat) (position address total : Nat)
    (heap : Mutable.Heap) (env : HeapCalls.Env)
    (found : heap[address]? = some (.tuple (numbers items))) (suffix : items.drop position = remaining)
    (positive : ∀ n ∈ remaining, 0 < n) (getTotal : env "product" = some (number total)) :
    BlockReturns program env
      [.sequenceStepLoop "position" (.ref address) position body [], .return (.name "product")]
      heap (number (remaining.foldl (fun total n => total * blockDigit 1 n) total)) heap := by
  have sequence : Mutable.sequence (.ref address) heap = some (.ok (numbers items), heap) := by
    simp [Mutable.sequence, found]
  induction remaining generalizing position total env with
  | nil =>
      apply BlockReturns.sequence_done (source := .ref address) sequence
      · have missing : items[position]? = none := by
          simpa [suffix] using (List.head?_drop (l := items) (i := position)).symm
        simp [numbers, missing]
      · exact BlockReturns.return_value (Evaluates.lookup getTotal)
  | cons head tail ih =>
      have item : items[position]? = some head := by
        simpa [suffix] using (List.head?_drop (l := items) (i := position)).symm
      let bound := setLocal env "position" (number head)
      let updated := setLocal bound "product" (number (total * blockDigit 1 head))
      have boundTotal : bound "product" = some (number total) := by simpa [bound, setLocal] using getTotal
      have tailSuffix : items.drop (position + 1) = tail := by simp [List.drop_add_one_eq_tail_drop, suffix]
      apply BlockReturns.sequence_step (source := .ref address) (value := number head) sequence (by simp [numbers, item])
      apply BlockReturns.augAssign (left := number total) (right := number (blockDigit 1 head))
        (value := number (total * blockDigit 1 head)) (middle := heap) (afterOperation := heap) boundTotal
      · exact Evaluates.call (.cons (Evaluates.lookup (show bound "position" = some (number head) from rfl)) .nil)
          (digit_at.result head (positive head (by simp)) heap)
      · simp [inplace, Mutable.binary, scalarBinary, number]
      · simpa only [Calls.loopContinuation, Calls.loopControl, Calls.loopControlStmt, Bool.false_or,
          Bool.false_eq_true, if_false, List.nil_append, List.foldl_cons] using
          ih (position + 1) (total * blockDigit 1 head) updated tailSuffix
            (fun n member => positive n (by simp [member])) rfl

/-- The optimized source handles any finite positive position tuple without allocating a stream. -/
theorem result (positions : List Nat) (positive : ∀ n ∈ positions, 0 < n) (heap : Mutable.Heap) (address : Nat)
    (found : heap[address]? = some (.tuple (numbers positions))) :
    ReturnsValue program "solve" [.ref address] heap (number (answer positions)) heap := by
  let initial := functionEnv ((findFunctionIn program.body "solve").getD ([], [])).1
    ((findFunctionIn program.body "solve").getD ([], [])).2 [.ref address] []
  let ready := setLocal initial "product" (number 1)
  apply BlockReturns.function rfl rfl
  apply BlockReturns.assign Evaluates.constant
  apply BlockReturns.for_sequence (items := numbers positions) (value := .ref address)
    (by intros; simp) (Evaluates.lookup (show ready "positions" = some (.ref address) from rfl))
    (by simp [Mutable.sequence, found])
  exact loop positions positions 0 address 1 heap ready found rfl positive rfl

end solve

private abbrev inputEnv (address : Nat) : HeapCalls.Env :=
  fun name => if name = "positions" then some (.ref address) else none

private theorem contents (heap : Mutable.Heap) (address : Nat)
    (typed : Mutable.Value.HasContainerType heap (.ref address) (.tupleOf (.scalar "int"))) :
    ∃ values : List Int, heap[address]? = some (.tuple (values.map (fun n => .scalar (.int n)))) := by
  obtain ⟨items, found, elements⟩ := Mutable.Value.hasContainerType_variadicTuple_iff.mp typed
  have projected : ∃ values : List Int, items = values.map (fun n => Mutable.Value.scalar (.int n)) := by
    clear found
    induction items with
    | nil => exact ⟨[], rfl⟩
    | cons first rest ih =>
        obtain ⟨n, rfl⟩ := Mutable.Value.hasContainerType_int_iff.mp (elements first (by simp))
        obtain ⟨values, rfl⟩ := ih (fun item member => elements item (List.mem_cons_of_mem _ member))
        exact ⟨n :: values, rfl⟩
  obtain ⟨values, rfl⟩ := projected
  exact ⟨values, found⟩

private theorem condition_result (values : List Int) (heap : Mutable.Heap) (address : Nat)
    (found : heap[address]? = some (.tuple (values.map (fun n => .scalar (.int n))))) :
    Evaluates program (inputEnv address) Solvers.P40Ast.requires_solve_0 heap
      (.scalar (.bool (decide (∀ n ∈ values, 0 < n)))) heap :=
  Evaluates.all_positive (name := "position") (sourceName := "positions")
    (Evaluates.lookup rfl) (by simp [Mutable.sequence, found])

private theorem domain (heap : Mutable.Heap) (address : Nat)
    (typed : Mutable.Value.HasContainerType heap (.ref address) (.tupleOf (.scalar "int")))
    (requires : HeapCalls.Condition program (inputEnv address) Solvers.P40Ast.requires_solve_0 heap) :
    ∃ positions : List Nat, (∀ n ∈ positions, 0 < n) ∧ heap[address]? = some (.tuple (numbers positions)) := by
  obtain ⟨values, found⟩ := contents heap address typed
  have positive : ∀ n ∈ values, 0 < n := by simpa using (condition_result values heap address found).condition_iff.mp requires
  refine ⟨values.map Int.toNat, ?_, ?_⟩
  · intro n member
    obtain ⟨value, present, rfl⟩ := List.mem_map.mp member
    have bound := positive value present
    omega
  · have encoded : numbers (values.map Int.toNat) = values.map (fun n => Mutable.Value.scalar (.int n)) := by
      simp only [numbers, List.map_map]
      apply List.map_congr_left
      intro value member
      simp only [Function.comp_apply, number, Int.toNat_of_nonneg (positive value member).le]
    rw [encoded]
    exact found

theorem requires_solve_0_supported (heap : Mutable.Heap) (address : Nat) (_valid : heap.Valid)
    (typed : Mutable.Value.HasContainerType heap (.ref address) (.tupleOf (.scalar "int"))) :
    (HeapCalls.evaluate program (inputEnv address) Solvers.P40Ast.requires_solve_0 heap).Supported := by
  obtain ⟨values, found⟩ := contents heap address typed
  exact Execution.supported_of_finishes (condition_result values heap address found).finishes

theorem solve.«Returns» (heap : Mutable.Heap) (address : Nat) (_valid : heap.Valid)
    (typed : Mutable.Value.HasContainerType heap (.ref address) (.tupleOf (.scalar "int")))
    (requires : HeapCalls.Condition program (inputEnv address) Solvers.P40Ast.requires_solve_0 heap) :
    (HeapCalls.call program "solve" [.ref address] heap).Returns := by
  obtain ⟨positions, positive, found⟩ := domain heap address typed requires
  exact (solve.result positions positive heap address found).returns

theorem solve.«Equiv(naive)» (heap : Mutable.Heap) (address : Nat) (valid : heap.Valid)
    (typed : Mutable.Value.HasContainerType heap (.ref address) (.tupleOf (.scalar "int")))
    (requires : HeapCalls.Condition program (inputEnv address) Solvers.P40Ast.requires_solve_0 heap) :
    (HeapCalls.call program "solve" [.ref address] heap).Equiv (HeapCalls.call P40Ast.module "naive" [.ref address] heap) := by
  obtain ⟨positions, positive, found⟩ := domain heap address typed requires
  obtain ⟨extra, reference⟩ :=
    data.«project-euler-statements».data.py.project_euler_statements.p40.naive.result positions positive heap address found
  apply Mutable.Call.equiv_of_finishes (solve.result positions positive heap address found).finishes reference.finishes
  simpa only [List.append_nil] using
    Mutable.Completion.equivalent_allocations heap [] extra (number (answer positions)) [] valid trivial (by simp)

theorem solve.«Supported» :
    (∀ (heap : Mutable.Heap) (address : Nat), heap.Valid →
      Mutable.Value.HasContainerType heap (.ref address) (.tupleOf (.scalar "int")) →
      HeapCalls.Condition program (inputEnv address) Solvers.P40Ast.requires_solve_0 heap →
      (HeapCalls.call program "solve" [.ref address] heap).Supported) ∧
    (∀ (heap : Mutable.Heap) (address : Nat), heap.Valid →
      Mutable.Value.HasContainerType heap (.ref address) (.tupleOf (.scalar "int")) →
      HeapCalls.Condition program (inputEnv address) Solvers.P40Ast.requires_solve_0 heap →
      (HeapCalls.call P40Ast.module "naive" [.ref address] heap).Supported) := by
  constructor
  · intro heap address _ typed requires
    obtain ⟨positions, positive, found⟩ := domain heap address typed requires
    exact (solve.result positions positive heap address found).returns.supported
  · intro heap address _ typed requires
    obtain ⟨positions, positive, found⟩ := domain heap address typed requires
    obtain ⟨_, reference⟩ :=
      data.«project-euler-statements».data.py.project_euler_statements.p40.naive.result positions positive heap address found
    exact reference.returns.supported

end solvers.«40»
