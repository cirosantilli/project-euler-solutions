import lang.py.FormalSW.IntegerParsing
import ProjectEulerStatements.Lib.ConcatenatedProductSearch
import data.«project-euler-statements».data.py.project_euler_statements.«p38.py»
import ProjectEulerStatements.Lib.PandigitalMask
import «.formalsw».solvers.«38.py.ast»
import lang.py.FormalSW.HeapCalls
import ProjectEulerStatements.Lib.ConcatenatedProductStrings

set_option Elab.async false

namespace solvers.«38»
open FormalSW.PyAst FormalSW.PyAst.HeapCalls
open ProjectEulerStatements.P38 ProjectEulerStatements.ConcatenatedProducts
private abbrev program := Solvers.P38Ast.module
private abbrev number (n : Int) : Mutable.Value := .scalar (.int n)
variable {globals : Mutable.Globals}

namespace concatenated_product
private abbrev code : List Stmt :=
  executableBody ((findFunctionIn program.body "concatenated_product").getD ([], [])).2
private abbrev condition : Expr := .compare (.name "total_len") [(.lt, .constant (.int 9))]
private abbrev body : List Stmt :=
  match (code[3]? : Option Stmt) with | some (.whileDo _ statements _) => statements | _ => []
private def values (x count : Nat) : List Mutable.Value :=
  (parts x count).map (fun piece => .scalar (.string piece))
private def collectorHeap (base : Mutable.Heap) (x count : Nat) : Mutable.Heap :=
  base ++ [.list (values x count)]
private def Locals (env : HeapCalls.Env) (address x count : Nat) : Prop :=
  env "x" = some (number x) ∧ env "n" = some (number count) ∧
    env "total_len" = some (number (numDigits (concatProduct x count))) ∧
    env "parts" = some (.ref address)
private def nextEnv (env : HeapCalls.Env) (x count : Nat) : HeapCalls.Env :=
  setLocal (setLocal (setLocal env "n" (number (count+1 : Nat))) "s"
    (.scalar (.string (toString ((x*(count+1) : Nat) : Int))))) "total_len"
    (number (numDigits (concatProduct x (count+1))))

private theorem iteration (x count : Nat) (positive : 0 < x)
    (base : Mutable.Heap) (env : HeapCalls.Env) (locals : Locals env base.length x count) :
    BlockContinues (globals := globals) program env body (collectorHeap base x count)
      (nextEnv env x count) (collectorHeap base x (count+1)) ∧
      Locals (nextEnv env x count) base.length x (count+1) := by
  let e1 := setLocal env "n" (number (count+1 : Nat))
  let piece := toString ((x*(count+1) : Nat) : Int)
  let e2 := setLocal e1 "s" (.scalar (.string piece))
  have first : BlockContinues (globals := globals) program env [body[0]'(by decide)]
      (collectorHeap base x count) e1 (collectorHeap base x count) :=
    BlockContinues.augAssign locals.2.1 Evaluates.constant
      (by simp [inplace, Mutable.binary, scalarBinary, number])
  have second : BlockContinues (globals := globals) program e1 [body[1]'(by decide)]
      (collectorHeap base x count) e2 (collectorHeap base x count) := by
    apply BlockContinues.assign
    apply Evaluates.builtin (.cons
      (Evaluates.binary (av := number x) (bv := number (count+1 : Nat))
        (value := number (x*(count+1) : Nat)) (middle := collectorHeap base x count)
        (afterRight := collectorHeap base x count) (finalHeap := collectorHeap base x count)
        (Evaluates.lookup (by simpa [e1, setLocal] using locals.1)) (Evaluates.lookup rfl)
        (by simp [Mutable.binary, scalarBinary, number])) .nil)
    rfl
  have third : BlockContinues (globals := globals) program e2 [body[2]'(by decide)]
      (collectorHeap base x count) e2 (collectorHeap base x (count+1)) := by
    apply BlockContinues.discard
    have appended := Evaluates.append (globals := globals) (module := program) (env := e2)
      (base := .name "parts") (expression := .name "s") (value := .scalar (.string piece))
      (heap := collectorHeap base x count)
      (Evaluates.lookup (by simpa [e2, e1, setLocal] using locals.2.2.2))
      (by simp [collectorHeap] : (collectorHeap base x count)[base.length]? = some (.list (values x count)))
      (Evaluates.lookup rfl)
      (by simp [collectorHeap] : (collectorHeap base x count)[base.length]? = some (.list (values x count)))
    simpa [collectorHeap, values, parts_succ, piece] using appended
  have fourth : BlockContinues (globals := globals) program e2 [body[3]'(by decide)]
      (collectorHeap base x (count+1)) (nextEnv env x count) (collectorHeap base x (count+1)) := by
    have lengthResult : Evaluates (globals := globals) program e2 (.builtinCall .len [.name "s"])
        (collectorHeap base x (count+1)) (number (numDigits (x*(count+1)))) (collectorHeap base x (count+1)) := by
      apply Evaluates.builtin (.cons (Evaluates.lookup rfl) .nil)
      have equal := ProjectEulerStatements.DecimalDigits.int_string_length (x*(count+1))
        (Nat.mul_pos positive (by omega))
      simpa [primitive, Mutable.length, number, numDigits, piece, equal]

    apply BlockContinues.augAssign
      (left := number (numDigits (concatProduct x count)))
      (right := number (numDigits (x*(count+1))))
      (value := number (numDigits (concatProduct x (count+1))))
      (middle := collectorHeap base x (count+1)) (finalHeap := collectorHeap base x (count+1))
      (by simpa [e2, e1, setLocal] using locals.2.2.1) lengthResult
    simp [inplace, Mutable.binary, scalarBinary, number, concatProduct_length_succ]
  refine ⟨first.append (second.append (third.append fourth)), ?_⟩
  rcases locals with ⟨getX, getN, getLength, getParts⟩
  exact ⟨by simpa [nextEnv, setLocal] using getX, by simp [nextEnv, setLocal],
    by simp [nextEnv, setLocal], by simpa [nextEnv, setLocal] using getParts⟩

private theorem loop (x count remaining : Nat) (positive : 0 < x)
    (endpoint : count+remaining = stopCount x) (base : Mutable.Heap) (env : HeapCalls.Env)
    (locals : Locals env base.length x count) :
    ∃ finalEnv, BlockContinues (globals := globals) program env [.whileDo condition body []]
      (collectorHeap base x count) finalEnv (collectorHeap base x (stopCount x)) ∧
      Locals finalEnv base.length x (stopCount x) := by
  induction remaining generalizing count env with
  | zero =>
      have same : count = stopCount x := by omega
      have stopped : ¬ numDigits (concatProduct x count) < 9 := by
        rw [same]
        have reached := stopCount_spec x positive
        omega
      have tested : Evaluates (globals := globals) program env condition (collectorHeap base x count)
          (.scalar (.bool false)) (collectorHeap base x count) :=
        Evaluates.compare (Evaluates.lookup locals.2.2.1) Evaluates.constant
          (by simp [HeapCalls.compare, scalarCompare, evalCmpOp, stopped])
      refine ⟨env, ?_, by simpa only [same] using locals⟩
      rw [← same]
      exact BlockContinues.while_loop tested rfl .nil
  | succ remaining ih =>
      have before : count < stopCount x := by omega
      have continues := before_stopCount x count positive before
      have tested : Evaluates (globals := globals) program env condition (collectorHeap base x count)
          (.scalar (.bool true)) (collectorHeap base x count) :=
        Evaluates.compare (Evaluates.lookup locals.2.2.1) Evaluates.constant
          (by simp [HeapCalls.compare, scalarCompare, evalCmpOp, continues])
      obtain ⟨iterated, nextLocals⟩ := iteration (globals := globals) x count positive base env locals
      obtain ⟨finalEnv, rest, finalLocals⟩ := ih (count+1) (by omega) (nextEnv env x count) nextLocals
      refine ⟨finalEnv, ?_, finalLocals⟩
      apply BlockContinues.while_loop (flag := true) tested rfl
      simpa [Calls.loopContinuation, Calls.loopControl, Calls.loopControlStmt] using iterated.append rest

/-- The original list-building loop returns its joined text and first sufficient factor count. -/
theorem result (x : Nat) (positive : 0 < x) (heap : Mutable.Heap) :
    ReturnsValue (globals := globals) program "concatenated_product" [number x] heap
      (.ref (heap.length+1))
      (heap ++ [.list (values x (stopCount x)),
        .tuple [.scalar (.string (toString (concatProduct x (stopCount x) : Int))), number (stopCount x)]]) := by
  let env := functionEnv ((findFunctionIn program.body "concatenated_product").getD ([], [])).1
    ((findFunctionIn program.body "concatenated_product").getD ([], [])).2 [number x] globals
  let e1 := setLocal env "parts" (.ref heap.length)
  let e2 := setLocal e1 "n" (number 0)
  let ready := setLocal e2 "total_len" (number 0)
  have initialLocals : Locals ready heap.length x 0 := by
    refine ⟨?_, rfl, rfl, rfl⟩
    cases globals <;> rfl
  obtain ⟨finalEnv, searched, finalLocals⟩ :=
    loop (globals := globals) x 0 (stopCount x) positive (by omega) heap ready initialLocals
  have joined : Evaluates (globals := globals) program finalEnv
      (.call (.attribute (.constant (.string "")) "join") [.name "parts"])
      (collectorHeap heap x (stopCount x))
      (.scalar (.string (toString (concatProduct x (stopCount x) : Int))))
      (collectorHeap heap x (stopCount x)) := by
    apply Evaluates.join Evaluates.constant (.cons (Evaluates.lookup finalLocals.2.2.2) .nil)
    · have operation := primitive_join_list (collectorHeap heap x (stopCount x)) heap.length ""
        (parts x (stopCount x)) (by simp [collectorHeap, values])
      simpa only [intercalate_parts, text_decimal x (stopCount x) positive
        (by have := (stopCount_bounds x positive).1; omega)] using operation
    · intro name impossible
      cases impossible
  apply BlockReturns.function rfl rfl
  apply BlockReturns.assign (value := .ref heap.length) (middle := heap ++ [.list []]) (Evaluates.list .nil)
  apply BlockReturns.assign Evaluates.constant
  apply BlockReturns.assign Evaluates.constant
  apply BlockContinues.returns searched
  apply BlockReturns.return_value
  have returned := Evaluates.tuple (globals := globals)
    (.cons joined (.cons (Evaluates.lookup finalLocals.2.1) .nil)) (by simp)
  simpa [collectorHeap, List.append_assoc] using returned

end concatenated_product
namespace is_1_to_9_pandigital
open FormalSW.PyAst FormalSW.PyAst.HeapCalls
open ProjectEulerStatements.PandigitalMask
private abbrev program := Solvers.P38Ast.module
private abbrev number (n : Nat) : Mutable.Value := .scalar (.int n)
private abbrev text (digits : List Nat) := String.ofList (digits.map Nat.digitChar)
private abbrev character (digit : Nat) : Mutable.Value :=
  .scalar (.string (String.singleton (Nat.digitChar digit)))
private abbrev code : List Stmt := executableBody
  ((findFunctionIn program.body "is_1_to_9_pandigital").getD ([], [])).2
private abbrev loopBody : List Stmt :=
  match (code[2]? : Option Stmt) with | some (.forIn _ _ body _) => body | _ => []
variable {globals : Mutable.Globals}

private theorem digit_ordinal (digit : Nat) (decimal : digit < 10) :
    ((Nat.digitChar digit).toNat : Int) - 48 = (digit : Int) := by
  interval_cases digit <;> decide

private theorem ordinal_evaluates (digit : Nat) (decimal : digit < 10)
    (heap : Mutable.Heap) (env : HeapCalls.Env)
    (getChar : env "ch" = some (character digit)) :
    Evaluates (globals := globals) program env
      (.binOp (.builtinCall .ord [.name "ch"]) .sub (.constant (.int 48)))
      heap (number digit) heap := by
  apply Evaluates.binary (av := number (Nat.digitChar digit).toNat) (bv := number 48)
    (middle := heap) (afterRight := heap)
  · apply Evaluates.builtin (.cons (Evaluates.lookup getChar) .nil)
    change scalarResult (some (Strings.ord (.string (String.ofList [Nat.digitChar digit])))) heap = _
    rw [Strings.ord_character]
    rfl
  · exact Evaluates.constant
  · simp [Mutable.binary, scalarBinary, number, digit_ordinal digit decimal]

/-- The sequence source is fixed; only the scalar mask local changes. Every
zero and repeated digit returns before the remaining source statements. -/
private theorem scan_loop (digits remaining : List Nat) (position mask : Nat)
    (suffix : digits.drop position = remaining)
    (decimal : ∀ digit ∈ remaining, digit < 10)
    (heap : Mutable.Heap) (env : HeapCalls.Env)
    (getMask : env "mask" = some (number mask))
    (getTarget : env "TARGET_MASK" = some (number 1022)) :
    BlockReturns (globals := globals) program env
      (.sequenceStepLoop "ch" (.scalar (.string (text digits))) position loopBody [] :: code.drop 3)
      heap (.scalar (.bool (ProjectEulerStatements.PandigitalMask.scan remaining mask))) heap := by
  have sequence : Mutable.sequence (.scalar (.string (text digits))) heap =
      some (.ok (digits.map character), heap) := by
    simp [Mutable.sequence, text, character, List.map_map]
  induction remaining generalizing position mask env with
  | nil =>
      apply BlockReturns.sequence_done (source := .scalar (.string (text digits))) sequence
      · have missing : digits[position]? = none := by
          simpa [suffix] using (List.head?_drop (l := digits) (i := position)).symm
        simp [missing]
      · apply BlockReturns.return_value
        apply Evaluates.compare (av := number mask) (bv := number 1022)
          (middle := heap) (afterRight := heap)
          (Evaluates.lookup getMask) (Evaluates.lookup getTarget)
        simp [HeapCalls.compare, scalarCompare, evalCmpOp, ProjectEulerStatements.PandigitalMask.scan]
        omega
  | cons head tail ih =>
      have item : digits[position]? = some head := by
        simpa [suffix] using (List.head?_drop (l := digits) (i := position)).symm
      have headDecimal := decimal head (by simp)
      have tailDecimal : ∀ digit ∈ tail, digit < 10 := by
        intro digit member
        exact decimal digit (by simp [member])
      let charEnv := setLocal env "ch" (character head)
      let digitEnv := setLocal charEnv "d" (number head)
      let bitEnv := setLocal digitEnv "bit" (number (2 ^ head))
      have getD : digitEnv "d" = some (number head) := by simp [digitEnv, setLocal]
      have getBit : bitEnv "bit" = some (number (2 ^ head)) := by simp [bitEnv, setLocal]
      have oldMask : bitEnv "mask" = some (number mask) := by
        simpa [bitEnv, digitEnv, charEnv, setLocal] using getMask
      apply BlockReturns.sequence_step (source := .scalar (.string (text digits)))
        (value := character head) sequence (by simp [item])
      apply BlockReturns.assign (ordinal_evaluates head headDecimal heap charEnv (by simp [charEnv, setLocal]))
      by_cases zero : head = 0
      · simp only [ProjectEulerStatements.PandigitalMask.scan, if_pos zero]
        apply BlockReturns.branch (flag := true)
          (Evaluates.compare (av := number head) (bv := number 0)
            (value := .scalar (.bool true)) (finalHeap := heap)
            (Evaluates.lookup getD) Evaluates.constant
            (by simp [HeapCalls.compare, scalarCompare, evalCmpOp, zero])) rfl
        exact BlockReturns.return_value Evaluates.constant
      · simp only [ProjectEulerStatements.PandigitalMask.scan, if_neg zero]
        apply BlockReturns.branch (flag := false)
          (Evaluates.compare (av := number head) (bv := number 0)
            (value := .scalar (.bool false)) (finalHeap := heap)
            (Evaluates.lookup getD) Evaluates.constant
            (by simp [HeapCalls.compare, scalarCompare, evalCmpOp, zero])) rfl
        apply BlockReturns.assign (value := number (2 ^ head)) (middle := heap)
        · apply Evaluates.binary (av := number 1) (bv := number head)
            (middle := heap) (afterRight := heap) Evaluates.constant (Evaluates.lookup getD)
          simp [Mutable.binary, scalarShiftLeft, integerShiftLeft, number,
            show ¬ (head : Int) < 0 by omega]
        · have masked : Evaluates (globals := globals) program bitEnv
              (.binOp (.name "mask") .bitAnd (.name "bit")) heap
              (number (mask &&& 2 ^ head)) heap := by
            exact Evaluates.binary (Evaluates.lookup oldMask) (Evaluates.lookup getBit)
              (by simp [Mutable.binary, scalarBitAnd, number]; rfl)
          by_cases overlap : mask &&& 2 ^ head ≠ 0
          · rw [if_pos overlap]
            apply BlockReturns.branch (flag := true) masked
              (by simp [Mutable.truth, scalarTruth, overlap])
            exact BlockReturns.return_value Evaluates.constant
          · rw [if_neg overlap]
            have empty : mask &&& 2 ^ head = 0 := by omega
            apply BlockReturns.branch (flag := false) masked
              (by simp [Mutable.truth, scalarTruth, empty])
            apply BlockReturns.augAssign (left := number mask) (right := number (2 ^ head))
              (value := number (mask ||| 2 ^ head)) (middle := heap) (afterOperation := heap)
              oldMask (Evaluates.lookup getBit)
            · simp [inplace, Mutable.binary, scalarBitOr, number]
              rfl
            · have rest := ih (position + 1) (mask ||| 2 ^ head)
                (by simp [List.drop_add_one_eq_tail_drop, suffix]) tailDecimal
                (setLocal bitEnv "mask" (number (mask ||| 2 ^ head)))
                (by simp [setLocal])
                (by simpa [setLocal, bitEnv, digitEnv, charEnv] using getTarget)
              simpa [loopBody, code, program, Solvers.P38Ast.module, findFunctionIn, executableBody,
                Calls.loopContinuation, Calls.loopControl, Calls.loopControlStmt,
                bitEnv, digitEnv, charEnv] using rest

/-- Decimal inputs have nonnegative shifts; the heap is unchanged on every path.
The broader Unicode-string input domain is deliberately not asserted here. -/
theorem result (digits : List Nat) (decimal : ∀ digit ∈ digits, digit < 10)
    (heap : Mutable.Heap) (globals : Mutable.Globals)
    (target : globals.lookup "TARGET_MASK" = some (number 1022)) :
    ReturnsValue (globals := globals) program "is_1_to_9_pandigital"
      [.scalar (.string (text digits))] heap (.scalar (.bool (check digits))) heap := by
  let initial := functionEnv ((findFunctionIn program.body "is_1_to_9_pandigital").getD ([], [])).1
    ((findFunctionIn program.body "is_1_to_9_pandigital").getD ([], [])).2
    [.scalar (.string (text digits))] globals
  have getInput : initial "s" = some (.scalar (.string (text digits))) := by cases globals <;> rfl
  have getTarget : initial "TARGET_MASK" = some (number 1022) := by
    cases globals with
    | nil => simp at target
    | cons head tail => exact target
  have lengthTest : Evaluates (globals := globals) program initial
      (.compare (.builtinCall .len [.name "s"]) [(.notEq, .constant (.int 9))])
      heap (.scalar (.bool (decide (digits.length ≠ 9)))) heap := by
    apply Evaluates.compare (av := number digits.length) (bv := number 9)
      (middle := heap) (afterRight := heap)
      (Evaluates.builtin (.cons (Evaluates.lookup getInput) .nil)
        (by simp [primitive, Mutable.length, text])) Evaluates.constant
    simp [HeapCalls.compare, scalarCompare, evalCmpOp]
    omega
  apply BlockReturns.function rfl rfl
  by_cases nine : digits.length = 9
  · simp only [check, nine, decide_true, Bool.true_and]
    apply BlockReturns.branch (flag := false) (by simpa only [nine, ne_eq, not_true_eq_false, decide_false] using lengthTest) rfl
    apply BlockReturns.assign Evaluates.constant
    apply BlockReturns.for_sequence (items := digits.map character) (by intros; simp)
      (Evaluates.lookup (by simpa [setLocal] using getInput))
      (by simp [Mutable.sequence, text, character, List.map_map])
    exact scan_loop digits digits 0 0 (by simp) decimal heap _
      (by simp [setLocal]) (by simpa [setLocal] using getTarget)
  · simp only [check, nine, decide_false, Bool.false_and]
    apply BlockReturns.branch (flag := true) (by simpa only [nine, ne_eq, not_false_eq_true, decide_true] using lengthTest) rfl
    exact BlockReturns.return_value Evaluates.constant
end is_1_to_9_pandigital

namespace solve
noncomputable section
private abbrev code : List Stmt :=
  executableBody ((findFunctionIn program.body "solve").getD ([], [])).2
private abbrev body : List Stmt :=
  match (code[1]? : Option Stmt) with | some (.forIn _ _ statements _) => statements | _ => []
private abbrev candidate (x : Nat) : Nat := concatProduct x (stopCount x)
private abbrev candidateText (x : Nat) : Mutable.Value := .scalar (.string (toString (candidate x : Int)))
private abbrev returnedValues (x : Nat) : List Mutable.Value := [candidateText x, number (stopCount x)]
private def extra (x : Nat) : Mutable.Heap :=
  [.list (concatenated_product.values x (stopCount x)), .tuple (returnedValues x)]
private abbrev guard : Expr := .boolOp .and
  [.compare (.name "n") [(.gt, .constant (.int 1))],
    .call (.name "is_1_to_9_pandigital") [.name "s"]]
private abbrev acceptedBody : List Stmt :=
  [.assign [.name "val"] (.builtinCall .int [.name "s"]),
    .ifThen (.compare (.name "val") [(.gt, .name "best")]) [.assign [.name "best"] (.name "val")] []]

private theorem pandigital_result (x : Nat) (positive : 0 < x) (heap : Mutable.Heap)
    (full : globals.lookup "TARGET_MASK" = some (number 1022)) :
    ReturnsValue (globals := globals) program "is_1_to_9_pandigital" [candidateText x] heap
      (.scalar (.bool (isPandigital1to9 (candidate x)))) heap := by
  have valuePositive := concatProduct_positive x (stopCount x) positive
    (by have := (stopCount_bounds x positive).1; omega)
  have printed : String.ofList ((digitsBE (candidate x)).map Nat.digitChar) =
      toString (candidate x : Int) := by
    apply String.toList_injective
    rw [String.toList_ofList, ProjectEulerStatements.ConcatenatedProducts.decimal_characters _ valuePositive]
  have result := is_1_to_9_pandigital.result (digitsBE (candidate x))
    (by intro d member; exact Nat.digits_lt_base (by decide) (List.mem_reverse.mp member)) heap globals full
  simpa only [is_1_to_9_pandigital.text, printed, ProjectEulerStatements.PandigitalMask.check_digitsBE] using result

private theorem guard_result (x : Nat) (positive : 0 < x) (heap : Mutable.Heap) (env : HeapCalls.Env)
    (getCount : env "n" = some (number (stopCount x))) (getText : env "s" = some (candidateText x))
    (full : globals.lookup "TARGET_MASK" = some (number 1022)) :
    Evaluates (globals := globals) program env guard heap (.scalar (.bool (accepted x (stopCount x)))) heap := by
  have tested : Evaluates (globals := globals) program env (.compare (.name "n") [(.gt, .constant (.int 1))])
      heap (.scalar (.bool (decide (1 < stopCount x)))) heap :=
    Evaluates.compare (Evaluates.lookup getCount) Evaluates.constant
      (by simp [HeapCalls.compare, scalarCompare, evalCmpOp])
  rw [accepted_eq]
  by_cases several : 1 < stopCount x
  · simp only [several, decide_true, Bool.true_and]
    apply Evaluates.boolean_next (flag := true) (by simpa [several] using tested) rfl rfl
    exact Evaluates.boolean_last
      (Evaluates.call (.cons (Evaluates.lookup getText) .nil) (pandigital_result x positive heap full)) rfl
  · simp only [several, decide_false, Bool.false_and]
    exact Evaluates.boolean_stop (flag := false) (by simpa [several] using tested) rfl rfl

private theorem accepted_result (x best : Nat) (heap : Mutable.Heap) (env : HeapCalls.Env)
    (getText : env "s" = some (candidateText x)) (getBest : env "best" = some (number best)) :
    ∃ finalEnv, BlockContinues (globals := globals) program env acceptedBody heap finalEnv heap ∧
      finalEnv "best" = some (number (max best (candidate x) : Nat)) := by
  let e1 := setLocal env "val" (number (candidate x))
  have parsed : Evaluates (globals := globals) program env (.builtinCall .int [.name "s"])
      heap (number (candidate x)) heap := by
    apply Evaluates.builtin (.cons (Evaluates.lookup getText) .nil)
    simp only [primitive, candidateText, List.mapM_cons, List.mapM_nil, pure_bind]
    rw [Calls.integerConversion_int]
    rfl
  have first : BlockContinues (globals := globals) program env
      [.assign [.name "val"] (.builtinCall .int [.name "s"])] heap e1 heap := BlockContinues.assign parsed
  have compared : Evaluates (globals := globals) program e1
      (.compare (.name "val") [(.gt, .name "best")]) heap
      (.scalar (.bool (decide (best < candidate x)))) heap :=
    Evaluates.compare (Evaluates.lookup rfl) (Evaluates.lookup (by simpa [e1, setLocal] using getBest))
      (by simp [HeapCalls.compare, scalarCompare, evalCmpOp])
  by_cases better : best < candidate x
  · let finalEnv := setLocal e1 "best" (number (candidate x))
    refine ⟨finalEnv, first.append ?_, ?_⟩
    · exact BlockContinues.branch (flag := true) (by simpa [better] using compared) rfl
        (BlockContinues.assign (Evaluates.lookup rfl))
    · simp [finalEnv, setLocal, max_eq_right (Nat.le_of_lt better)]
  · refine ⟨e1, first.append ?_, ?_⟩
    · exact BlockContinues.branch (flag := false) (by simpa [better] using compared) rfl .nil
    · simpa [e1, setLocal, max_eq_left (Nat.le_of_not_gt better)] using getBest

private theorem iteration (x best : Nat) (positive : 0 < x) (heap : Mutable.Heap) (env : HeapCalls.Env)
    (getX : env "x" = some (number x)) (getBest : env "best" = some (number best))
    (full : globals.lookup "TARGET_MASK" = some (number 1022)) :
    ∃ finalEnv, BlockContinues (globals := globals) program env body heap finalEnv (heap ++ extra x) ∧
      finalEnv "best" = some (number (max best (seedScore x) : Nat)) := by
  let unpacked := assignNames env ["s", "n"] (returnedValues x)
  let after := heap ++ extra x
  have first : BlockContinues (globals := globals) program env [body[0]'(by decide)] heap unpacked after := by
    apply BlockContinues.unpack (names := ["s", "n"]) (items := returnedValues x) rfl
      (Evaluates.call (.cons (Evaluates.lookup getX) .nil) (concatenated_product.result x positive heap))
    · simp [Mutable.sequence, after, extra]
    · rfl
  have getText : unpacked "s" = some (candidateText x) := by simp [unpacked, returnedValues, assignNames, setLocal]
  have getCount : unpacked "n" = some (number (stopCount x)) := by simp [unpacked, returnedValues, assignNames, setLocal]
  have sameBest : unpacked "best" = some (number best) := by simpa [unpacked, returnedValues, assignNames, setLocal] using getBest
  have tested := guard_result x positive after unpacked getCount getText full
  cases yes : accepted x (stopCount x) with
  | false =>
      refine ⟨unpacked, first.append ?_, ?_⟩
      · exact BlockContinues.branch (flag := false) (by simpa [yes] using tested) rfl .nil
      · simpa only [seedScore, yes, Bool.false_eq_true, if_false, Nat.max_zero] using sameBest
  | true =>
      obtain ⟨finalEnv, executed, finalBest⟩ := accepted_result x best after unpacked getText sameBest
      refine ⟨finalEnv, first.append ?_, ?_⟩
      · exact BlockContinues.branch (flag := true) (by simpa [yes] using tested) rfl executed
      · simpa only [seedScore, yes, if_true] using finalBest

private theorem loop (first count best : Nat) (positive : 0 < first) (endpoint : first+count=10000)
    (heap : Mutable.Heap) (env : HeapCalls.Env) (getBest : env "best" = some (number best))
    (full : globals.lookup "TARGET_MASK" = some (number 1022)) :
    ∃ finalHeap, BlockReturns (globals := globals) program env
      [.rangeStepLoop "x" first 10000 1 body [], .return (.name "best")]
      heap (number (scan first count best)) finalHeap := by
  induction count generalizing first best heap env with
  | zero =>
      refine ⟨heap, ?_⟩
      rw [scan_zero]
      apply BlockReturns.range_done (by simp [Calls.rangeContinues]; omega)
      exact BlockReturns.return_value (Evaluates.lookup getBest)
  | succ count ih =>
      let bound := setLocal env "x" (number first)
      obtain ⟨middleEnv, iterated, middleBest⟩ := iteration first best positive heap bound
        (by simp [bound, setLocal]) (by simpa [bound, setLocal] using getBest) full
      obtain ⟨finalHeap, rest⟩ := ih (first+1) (max best (seedScore first)) (by omega) (by omega)
        (heap ++ extra first) middleEnv middleBest
      refine ⟨finalHeap, ?_⟩
      rw [scan_succ]
      apply BlockReturns.range_step (by simp [Calls.rangeContinues]; omega)
      apply BlockContinues.returns iterated
      simpa [Calls.loopContinuation, Calls.loopControl, Calls.loopControlStmt] using rest

/-- Source-level running maximum, preserving its calls, tuple unpacking, and short-circuit guard. -/
theorem result (heap : Mutable.Heap) (globals : Mutable.Globals)
    (full : globals.lookup "TARGET_MASK" = some (.scalar (.int 1022))) :
    ∃ finalHeap, ReturnsValue (globals := globals) program "solve" [] heap
      (number (ProjectEulerStatements.P38.naive 9999)) finalHeap := by
  let env := functionEnv ((findFunctionIn program.body "solve").getD ([], [])).1
    ((findFunctionIn program.body "solve").getD ([], [])).2 [] globals
  let ready := setLocal env "best" (number 0)
  obtain ⟨finalHeap, searched⟩ := loop (globals := globals) 1 9999 0 (by decide) rfl heap ready rfl full
  refine ⟨finalHeap, ?_⟩
  rw [reference_scan]
  apply BlockReturns.function rfl rfl
  apply BlockReturns.assign Evaluates.constant
  apply BlockReturns.for_range (.cons Evaluates.constant (.cons Evaluates.constant .nil)) rfl
  exact searched

end
end solve

private abbrev contractGlobals (mask : Int) : Mutable.Globals :=
  [("TARGET_MASK", .scalar (.int mask))]
private abbrev contractEnv (mask : Int) : HeapCalls.Env :=
  fun name => (contractGlobals mask).lookup name

private theorem condition_result (mask : Int) :
    Evaluates (globals := contractGlobals mask) program (contractEnv mask)
      Solvers.P38Ast.requires_solve_0 [] (.scalar (.bool (decide (mask = 1022)))) [] := by
  exact Evaluates.compare (av := number mask) (bv := number 1022)
    (Evaluates.lookup rfl) Evaluates.constant
    (by simp [HeapCalls.compare, scalarCompare, evalCmpOp])

private theorem accepted_mask (mask : Int)
    (condition : HeapCalls.Condition program (contractEnv mask) Solvers.P38Ast.requires_solve_0 []
      (contractGlobals mask)) : mask = 1022 := by
  simpa using (condition_result mask).condition_iff.mp condition

theorem requires_solve_0_supported (mask : Int) :
    (HeapCalls.evaluate program (contractEnv mask) Solvers.P38Ast.requires_solve_0 []
      (contractGlobals mask)).Supported :=
  Execution.supported_of_finishes (condition_result mask).finishes

theorem solve.«Equiv(naive_default)» (mask : Int)
    (condition : HeapCalls.Condition program (contractEnv mask) Solvers.P38Ast.requires_solve_0 []
      (contractGlobals mask)) :
    (HeapCalls.call program "solve" [] [] (contractGlobals mask)).Equiv
      (HeapCalls.call P38Ast.module "naive_default" [] [] (contractGlobals mask)) := by
  have same := accepted_mask mask condition
  subst mask
  obtain ⟨leftHeap, optimized⟩ := solve.result [] (contractGlobals 1022) rfl
  obtain ⟨rightHeap, reference⟩ :=
    data.«project-euler-statements».data.py.project_euler_statements.p38.naive_default.result
      (globals := contractGlobals 1022) []
  apply Mutable.Call.equiv_of_finishes optimized.finishes reference.finishes
  exact Mutable.Completion.equivalent_allocations [] leftHeap rightHeap
    (number (ProjectEulerStatements.P38.naive 9999)) (contractGlobals 1022)
    (by simp [Mutable.Heap.Valid]) trivial
    (by simp [contractGlobals, Mutable.Value.edge, Graph.Edge.Within])

theorem solve.«Supported» :
    (∀ mask : Int, HeapCalls.Condition program (contractEnv mask) Solvers.P38Ast.requires_solve_0 []
      (contractGlobals mask) → (HeapCalls.call program "solve" [] [] (contractGlobals mask)).Supported) ∧
    (∀ mask : Int, HeapCalls.Condition program (contractEnv mask) Solvers.P38Ast.requires_solve_0 []
      (contractGlobals mask) →
        (HeapCalls.call P38Ast.module "naive_default" [] [] (contractGlobals mask)).Supported) := by
  constructor
  · intro mask condition
    have same := accepted_mask mask condition
    subst mask
    obtain ⟨_, optimized⟩ := solve.result [] (contractGlobals 1022) rfl
    exact Execution.supported_of_finishes optimized.finishes
  · intro mask _
    obtain ⟨_, reference⟩ :=
      data.«project-euler-statements».data.py.project_euler_statements.p38.naive_default.result
        (globals := contractGlobals mask) []
    exact Execution.supported_of_finishes reference.finishes

end solvers.«38»
