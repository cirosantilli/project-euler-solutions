import «.formalsw».solvers.«40.py.ast»
import lang.py.FormalSW.HeapCalls
import lang.py.FormalSW.IntegerParsing
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
namespace solve
private def positions : List Nat := [1, 10, 100, 1000, 10000, 100000, 1000000]
private def numbers (items : List Nat) : List Mutable.Value := items.map number
private abbrev loopBody : List Stmt :=
  [.augAssign (.name "prod") .mult (.call (.name "digit_at") [.name "p"])]

private theorem loop (items remaining : List Nat) (position address total limit : Nat)
    (heap : Mutable.Heap) (env : HeapCalls.Env)
    (found : heap[address]? = some (.list (numbers items)))
    (suffix : items.drop position = remaining)
    (bounds : ∀ n ∈ remaining, 0 < n ∧ n ≤ limit)
    (getTotal : env "prod" = some (number total)) :
    BlockReturns program env
      [.sequenceStepLoop "p" (.ref address) position loopBody [], .return (.name "prod")]
      heap (number (remaining.foldl (fun acc n => acc * P40.digitAt n limit) total)) heap := by
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
      have headBounds := bounds head (by simp)
      let bound := setLocal env "p" (number head)
      have getP : bound "p" = some (number head) := by simp [bound, setLocal]
      have boundTotal : bound "prod" = some (number total) := by
        simpa [bound, setLocal] using getTotal
      have tailSuffix : items.drop (position+1) = tail := by
        simp [List.drop_add_one_eq_tail_drop, suffix]
      have tailBounds : ∀ n ∈ tail, 0 < n ∧ n ≤ limit := fun n member => bounds n (by simp [member])
      have digit : Evaluates program bound (.call (.name "digit_at") [.name "p"])
          heap (number (P40.digitAt head limit)) heap := by
        have evaluated := Evaluates.call (module := program) (env := bound)
          (.cons (Evaluates.lookup getP) .nil) (digit_at.result head headBounds.1 heap)
        simpa only [blockDigit_correct head limit headBounds.1 headBounds.2] using evaluated
      apply BlockReturns.sequence_step (source := .ref address) (value := number head)
        sequence (by simp [numbers, item])
      apply BlockReturns.augAssign (left := number total) (right := number (P40.digitAt head limit))
        (value := number (total*P40.digitAt head limit)) (middle := heap) (afterOperation := heap)
        boundTotal digit
      · simp [inplace, Mutable.binary, scalarBinary, number]
      · simpa only [Calls.loopContinuation, Calls.loopControl, Calls.loopControlStmt,
          Bool.false_or, Bool.false_eq_true, if_false, List.nil_append, List.foldl_cons] using
          ih (position+1) (total*P40.digitAt head limit)
            (setLocal bound "prod" (number (total*P40.digitAt head limit))) tailSuffix tailBounds rfl

/-- Allocate the seven source positions and multiply their independently computed digits. -/
theorem result (heap : Mutable.Heap) :
    ReturnsValue program "solve" [] heap (number (P40.naive 1000000))
      (heap ++ [.list (numbers positions)]) := by
  apply BlockReturns.function rfl rfl
  apply BlockContinues.returns (BlockContinues.annAssign (value := .ref heap.length)
    (Evaluates.list (.cons Evaluates.constant (.cons Evaluates.constant (.cons Evaluates.constant
      (.cons Evaluates.constant (.cons Evaluates.constant (.cons Evaluates.constant
        (.cons Evaluates.constant .nil)))))))))
  apply BlockReturns.assign Evaluates.constant
  apply BlockReturns.for_sequence (items := numbers positions) (value := .ref heap.length)
    (by intros; simp) (Evaluates.lookup (by simp [setLocal])) (by simp [Mutable.sequence, numbers, positions])
  exact loop positions positions 0 heap.length 1 1000000 _ _ (by simp [numbers, positions]) rfl
    (by intro n member; simp only [positions, List.mem_cons, List.not_mem_nil, or_false] at member; omega)
    (by simp [setLocal])

end solve

theorem solve.«Equiv(naive_default)» :
    (HeapCalls.scalarFunction program "solve" []).Equiv
      (HeapCalls.scalarFunction P40Ast.module "naive_default" []) := by
  obtain ⟨_, reference⟩ :=
    data.«project-euler-statements».data.py.project_euler_statements.p40.naive_default.result []
  have optimized := HeapCalls.ReturnsValue.scalar_finishes (values := []) (solve.result [])
  have original := HeapCalls.ReturnsValue.scalar_finishes (values := []) reference
  intro outcome
  constructor
  · intro finished
    rw [finished.unique optimized]
    exact original
  · intro finished
    rw [finished.unique original]
    exact optimized

theorem solve.«Supported» :
    (HeapCalls.scalarFunction program "solve" []).Supported ∧
      (HeapCalls.scalarFunction P40Ast.module "naive_default" []).Supported :=
  ⟨(HeapCalls.ReturnsValue.scalar_returns (values := []) (solve.result [])).supported,
    data.«project-euler-statements».data.py.project_euler_statements.p40.naive_default.«Supported»⟩

end solvers.«40»
