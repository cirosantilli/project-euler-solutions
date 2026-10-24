import «.formalsw».solvers.«42.py.ast»
import data.«project-euler-statements».data.py.project_euler_statements.«p42.py»
import lang.py.FormalSW.TruthCalls
import ProjectEulerSolutions.Lib.FilteredSum
import ProjectEulerSolutions.Lib.TriangularDiscriminant
import Mathlib.Tactic.IntervalCases

set_option Elab.async false

namespace solvers.«42»
open FormalSW.PyAst FormalSW.PyAst.HeapCalls
open ProjectEulerSolutions.TriangularDiscriminant
open data.«project-euler-statements».data.py.project_euler_statements.p42 (letter score triangularTest count)
private abbrev program := Solvers.P42Ast.module
private abbrev number (n : Int) : Mutable.Value := .scalar (.int n)
private abbrev flag (b : Bool) : Mutable.Value := .scalar (.bool b)
private abbrev character (c : Char) : Mutable.Value := .scalar (.string (String.singleton c))
private abbrev string (s : String) : Mutable.Value := .scalar (.string s)

def upper (c : Char) : Bool := decide (65 ≤ c.toNat ∧ c.toNat ≤ 90)
def uppercase (word : String) : Bool := !word.toList.isEmpty && word.toList.all upper

namespace is_triangle
/-- The modeled integer square root satisfies the same mathematical bounds as Nat.sqrt. -/
theorem result (n : Nat) (positive : 0 < n) (heap : Mutable.Heap) :
    ReturnsValue program "is_triangle" [number n] heap
      (flag (ProjectEulerStatements.P42.isTriangle n)) heap := by
  let initial := functionEnv ((findFunctionIn program.body "is_triangle").getD ([], [])).1
    ((findFunctionIn program.body "is_triangle").getD ([], [])).2 [number n] []
  let d := 1+8*n
  let root := Nat.sqrt d
  let e1 := setLocal initial "d" (number d)
  let e2 := setLocal e1 "s" (number root)
  have rootPositive : 0 < root := Nat.sqrt_pos.mpr (by dsimp [d]; omega)
  have assignedD : Evaluates program initial
      (.binOp (.constant (.int 1)) .add (.binOp (.constant (.int 8)) .mult (.name "x")))
      heap (number d) heap := by
    apply Evaluates.binary (middle := heap) (afterRight := heap) (finalHeap := heap)
      (av := number 1) (bv := number (8*n)) Evaluates.constant
    · exact Evaluates.binary (middle := heap) (afterRight := heap) (finalHeap := heap)
        Evaluates.constant (Evaluates.lookup rfl) (by simp [Mutable.binary, scalarBinary, number])
    · simp [Mutable.binary, scalarBinary, number, d]
  have assignedRoot : Evaluates program e1 (.call (.attribute (.name "math") "isqrt") [.name "d"])
      heap (number root) heap := by
    apply Evaluates.qualified_call (by decide) (.cons (Evaluates.lookup rfl) .nil)
    intro anchors stack
    apply Reaches.next (following := ⟨.value (number root), stack, heap⟩)
    · change HeapCalls.result anchors heap (scalarResult (some (Math.isqrtCall [.int d]))) stack = _
      have same : Math.isqrt d = Nat.sqrt d := Nat.eq_sqrt.mpr (Math.isqrt_spec d)
      simp [HeapCalls.result, run, scalarResult, Math.isqrtCall, same, root, number,
        show ¬ (d : Int) < 0 by omega]
    · exact .refl
  have tested : Evaluates program e2
      (.compare (.binOp (.name "s") .mult (.name "s")) [(.notEq, .name "d")])
      heap (flag (decide (root*root ≠ d))) heap := by
    apply Evaluates.compare (middle := heap) (afterRight := heap) (av := number (root*root)) (bv := number d)
    · exact Evaluates.binary (middle := heap) (afterRight := heap) (finalHeap := heap)
        (Evaluates.lookup rfl) (Evaluates.lookup rfl) (by simp [Mutable.binary, scalarBinary, number])
    · exact Evaluates.lookup (by simp [e2, e1, setLocal])
    · change some (Except.ok (ε := ExceptionObject) (flag (decide (((root*root : Nat) : Int) ≠ (d : Int)))), heap) = _
      simp only [ne_eq, Int.natCast_inj]
  rw [← correct n positive, discriminant, if_neg (by omega : n ≠ 0)]
  apply BlockReturns.function rfl rfl
  apply BlockReturns.branch (middle := heap) (flag := false) (value := flag false)
    (Evaluates.compare (middle := heap) (afterRight := heap) (av := number n) (bv := number 0)
      (Evaluates.lookup rfl) Evaluates.constant
      (by simp [HeapCalls.compare, scalarCompare, evalCmpOp]; omega)) rfl
  apply BlockReturns.assign assignedD
  apply BlockReturns.assign assignedRoot
  by_cases square : root*root = d
  · simp only [show Nat.sqrt (1+8*n)*Nat.sqrt (1+8*n)=1+8*n from square, true_and]
    apply BlockReturns.branch (flag := false) tested
    · simp [Mutable.truth, scalarTruth, square]
    · apply BlockReturns.return_value
      apply Evaluates.compare (middle := heap) (afterRight := heap)
        (av := number (((root-1)%2 : Nat) : Int)) (bv := number 0)
      · apply Evaluates.binary (middle := heap) (afterRight := heap) (finalHeap := heap)
          (av := number ((root-1 : Nat) : Int)) (bv := number 2)
        · exact Evaluates.binary (middle := heap) (afterRight := heap) (finalHeap := heap)
            (Evaluates.lookup rfl) Evaluates.constant
            (by simp [Mutable.binary, scalarBinary, number, Int.natCast_sub (by omega : 1 ≤ root)])
        · exact Evaluates.constant
        · simp [Mutable.binary, scalarBinary, number]
      · exact Evaluates.constant
      · change some (Except.ok (ε := ExceptionObject) (flag (decide ((((root-1)%2 : Nat) : Int)=0))), heap) = _
        simp only [Int.natCast_eq_zero, root, d]
  · simp only [show ¬ Nat.sqrt (1+8*n)*Nat.sqrt (1+8*n)=1+8*n from square, false_and, decide_false]
    apply BlockReturns.branch (flag := true) tested
    · simp [Mutable.truth, scalarTruth, square]
    · exact BlockReturns.return_value Evaluates.constant
end is_triangle

namespace word_value
private theorem ascii_comparisons (c : Char) (bounds : 65 ≤ c.toNat ∧ c.toNat ≤ 90) :
    scalarCompare .ltE (.string "A") (.string (String.singleton c)) = some true ∧
    scalarCompare .ltE (.string (String.singleton c)) (.string "Z") = some true := by
  have recovered := (Char.ofNat_toNat c).symm
  obtain ⟨lowerBound, upperBound⟩ := bounds
  interval_cases c.toNat <;> subst c <;> decide +kernel

private theorem filter_result (c : Char) (bounds : 65 ≤ c.toNat ∧ c.toNat ≤ 90)
    (heap : Mutable.Heap) (env : HeapCalls.Env) (getC : env "c" = some (character c)) :
    Evaluates program env
      (.compare (.constant (.string "A")) [(.ltE, .name "c"), (.ltE, .constant (.string "Z"))])
      heap (flag true) heap := by
  obtain ⟨lower, upper⟩ := ascii_comparisons c bounds
  apply Evaluates.chain Evaluates.constant
  apply Evaluates.Compares.next (compared := flag true) (afterComparison := heap) (Evaluates.lookup getC)
  · simp [HeapCalls.compare, lower]
  · rfl
  · apply Evaluates.Compares.last (value := string "Z") (middle := heap) Evaluates.constant
    simp [HeapCalls.compare, upper]

/-- Each retained uppercase character has exactly its raw reference ordinal value. -/
theorem result (word : String) (accepted : word.toList.all upper = true) (heap : Mutable.Heap) :
    ReturnsValue program "word_value" [string word] heap (number (score word)) heap := by
  let characters := word.toList.attach
  have folded : characters.foldl (fun total c => total + letter c.val) 0 = score word := by
    simp [characters, score]
  have sequence : Mutable.sequence (string word) heap =
      some (.ok (characters.map (fun c => character c.val)), heap) := by
    simp [Mutable.sequence, characters]
  apply BlockReturns.function rfl rfl
  apply BlockReturns.return_value
  rw [← folded]
  apply Evaluates.sum_generator_sequence (items := characters.map (fun c => character c.val))
    (by intros; simp) (Evaluates.lookup rfl) sequence Evaluates.constant rfl
  have fold : characters.foldl (fun total c => if true then total+letter c.val else total) 0 =
      characters.foldl (fun total c => total+letter c.val) 0 := rfl
  rw [← fold]
  apply ProjectEulerSolutions.FilteredSum.loop (source := .scalar (.string word)) (heap := heap) (values := characters)
    (encode := fun c => character c.val) (predicate := fun _ => true) (number := fun c => letter c.val)
    characters (by decide) sequence
  · intro env c getC
    have bounds := List.all_eq_true.mp accepted c.val c.property
    exact filter_result c.val (of_decide_eq_true bounds) heap env getC
  · intro env c getC _
    apply Evaluates.binary (middle := heap) (afterRight := heap) (finalHeap := heap)
      (av := number ((c.val.toNat : Int)-65)) (bv := number 1)
    · apply Evaluates.binary (middle := heap) (afterRight := heap) (finalHeap := heap)
        (av := number c.val.toNat) (bv := number 65)
      · apply Evaluates.builtin (.cons (Evaluates.lookup getC) .nil)
        change scalarResult (some (Strings.ord (.string (String.singleton c.val)))) heap = _
        rw [String.singleton_eq_ofList, Strings.ord_character]
        rfl
      · exact Evaluates.builtin (.cons Evaluates.constant .nil) rfl
      · rfl
    · exact Evaluates.constant
    · rfl
  · rfl
  · simp [setLocal]
end word_value

namespace Domain
private def test : Expr := .unaryOp .not
  (.compare (.constant (.int 65)) [(.ltE, .builtinCall .ord [.name "c"]), (.ltE, .constant (.int 90))])
private def characterBody : List Stmt := [.ifThen test [.return (.constant (.bool false))] []]
private abbrev emptyTest : Expr := .compare (.builtinCall .len [.name "word"]) [(.eq, .constant (.int 0))]
private def nameBody : List Stmt :=
  [.ifThen emptyTest [.return (.constant (.bool false))] [],
    .forIn (.name "c") (.name "word") characterBody []]

private theorem test_result (c : Char) (env : Mutable.Env) (heap : Mutable.Heap)
    (lookup : env "c" = some (.scalar (.string (String.singleton c)))) :
    Evaluates program env test heap (.scalar (.bool (!(upper c)))) heap := by
  apply Evaluates.unary (value := .scalar (.bool (upper c))) (middle := heap)
  · apply Evaluates.chain Evaluates.constant
    have ordinal : Evaluates program env (.builtinCall .ord [.name "c"]) heap (number c.toNat) heap := by
      apply Evaluates.builtin (.cons (Evaluates.lookup lookup) .nil)
      change scalarResult (some (Strings.ord (.string (String.ofList [c])))) heap = _
      rw [Strings.ord_character]
      rfl
    by_cases lower : 65 ≤ c.toNat
    · apply Evaluates.Compares.next ordinal (compared := .scalar (.bool true)) (afterComparison := heap)
      · simp [HeapCalls.compare, scalarCompare, evalCmpOp]
        omega
      · rfl
      · apply Evaluates.Compares.last (value := number 90) (middle := heap) Evaluates.constant
        simp [HeapCalls.compare, scalarCompare, evalCmpOp, upper, lower]
    · have rejected : upper c = false := by simp [upper, lower]
      rw [rejected]
      apply Evaluates.Compares.stop (middle := heap) (finalHeap := heap) ordinal
      · simp [HeapCalls.compare, scalarCompare, evalCmpOp]
        omega
      · rfl
  · rfl

/-- Character validation may return early, before any following statements execute. -/
private theorem characters_result (text : String) (remaining : List Char) (heap : Mutable.Heap)
    (position : Nat) (suffix : text.toList.drop position = remaining) (env : Mutable.Env)
    (following : List Stmt) (flag : Bool)
    (rest : ∀ env, BlockReturns program env following heap (.scalar (.bool flag)) heap) :
    BlockReturns program env
      (.sequenceStepLoop "c" (.scalar (.string text)) position characterBody [] :: following) heap
      (.scalar (.bool (remaining.all upper && flag))) heap := by
  have sequence : Mutable.sequence (.scalar (.string text)) heap =
      some (.ok (text.toList.map (fun c => .scalar (.string (String.singleton c)))), heap) := by
    simp [Mutable.sequence]
  induction remaining generalizing position env with
  | nil =>
      apply BlockReturns.sequence_done (source := .scalar (.string text)) (heap := heap) sequence
      · have missing : text.toList[position]? = none := by
          simpa [suffix] using (List.head?_drop (l := text.toList) (i := position)).symm
        simp [missing]
      · exact rest env
  | cons head tail ih =>
      have item : text.toList[position]? = some head := by
        simpa [suffix] using (List.head?_drop (l := text.toList) (i := position)).symm
      apply BlockReturns.sequence_step (source := .scalar (.string text)) (heap := heap)
        (value := .scalar (.string (String.singleton head))) sequence (by simp [item])
      have checked := test_result head (setLocal env "c" (.scalar (.string (String.singleton head)))) heap (by simp [setLocal])
      cases accepted : upper head with
      | false =>
          simp only [List.all_cons, accepted, Bool.false_and]
          apply BlockReturns.branch checked (flag := true) (by simp [Mutable.truth, scalarTruth, accepted])
          exact BlockReturns.return_value Evaluates.constant
      | true =>
          simp only [List.all_cons, accepted, Bool.true_and]
          apply BlockReturns.branch checked (flag := false) (by simp [Mutable.truth, scalarTruth, accepted])
          simpa [Calls.loopContinuation, Calls.loopControl, Calls.loopControlStmt, characterBody] using ih (position + 1)
            (by simp [List.drop_add_one_eq_tail_drop, suffix])
            (setLocal env "c" (.scalar (.string (String.singleton head))))


private theorem empty_result (word : String) (heap : Mutable.Heap) (env : HeapCalls.Env)
    (getWord : env "word" = some (string word)) :
    Evaluates program env emptyTest heap (flag (decide (word.toList = []))) heap := by
  apply Evaluates.compare (middle := heap) (afterRight := heap)
    (av := number word.toList.length) (bv := number 0)
  · apply Evaluates.builtin (.cons (Evaluates.lookup getWord) .nil)
    simp [primitive, Mutable.length]
  · exact Evaluates.constant
  · simp [HeapCalls.compare, scalarCompare, evalCmpOp]

private theorem words_result (words remaining : List String) (heap : Mutable.Heap) (address position : Nat)
    (found : heap[address]? = some (.list (words.map string)))
    (suffix : words.drop position = remaining) (env : HeapCalls.Env) (following : List Stmt) (value : Bool)
    (rest : ∀ env, BlockReturns program env following heap (flag value) heap) :
    BlockReturns program env (.sequenceStepLoop "word" (.ref address) position nameBody [] :: following) heap
      (flag (remaining.all uppercase && value)) heap := by
  have sequence : Mutable.sequence (.ref address) heap = some (.ok (words.map string), heap) := by
    simp [Mutable.sequence, found]
  induction remaining generalizing position env with
  | nil =>
      apply BlockReturns.sequence_done (source := .ref address) (heap := heap) sequence
      · have missing : words[position]? = none := by
          simpa [suffix] using (List.head?_drop (l := words) (i := position)).symm
        simp [missing]
      · exact rest env
  | cons head tail ih =>
      have item : words[position]? = some head := by
        simpa [suffix] using (List.head?_drop (l := words) (i := position)).symm
      let bound := setLocal env "word" (string head)
      have checked := empty_result head heap bound rfl
      apply BlockReturns.sequence_step (source := .ref address) (heap := heap)
        (value := string head) sequence (by simp [item])
      by_cases empty : head.toList = []
      · simp only [List.all_cons, uppercase, empty, List.isEmpty_nil, Bool.not_true,
          Bool.false_and]
        apply BlockReturns.branch (flag := true) checked
        · simp [Mutable.truth, scalarTruth, empty]
        · exact BlockReturns.return_value Evaluates.constant
      · have nonempty : head.toList.isEmpty = false := List.isEmpty_eq_false_iff.mpr empty
        apply BlockReturns.branch (flag := false) checked
        · simp [Mutable.truth, scalarTruth, empty]
        · apply BlockReturns.for_sequence (value := string head) (middle := heap)
            (items := head.toList.map character) (by intros; simp)
            (Evaluates.lookup rfl) (by simp [Mutable.sequence])
          have next := fun env => ih (position+1) (by simp [List.drop_add_one_eq_tail_drop, suffix]) env
          simpa [Calls.loopContinuation, Calls.loopControl, Calls.loopControlStmt, nameBody, characterBody,
            List.all_cons, uppercase, nonempty, Bool.and_assoc] using
            characters_result head head.toList heap 0 rfl bound
              (.sequenceStepLoop "word" (.ref address) (position+1) nameBody [] :: following)
              (tail.all uppercase && value) next
end Domain

/-- The executable prerequisite terminates for every typed list, including invalid words. -/
theorem words_are_uppercase.result (words : List String) (heap : Mutable.Heap) (address : Nat)
    (found : heap[address]? = some (.list (words.map string))) :
    ReturnsValue program "words_are_uppercase" [.ref address] heap (flag (words.all uppercase)) heap := by
  let definition := (findFunctionIn program.body "words_are_uppercase").getD ([], [])
  let entry := functionEnv definition.1 definition.2 [.ref address] []
  apply BlockReturns.function (args := definition.1) (body := definition.2) rfl rfl
  apply BlockReturns.for_sequence (items := words.map string) (middle := heap) (by intros; simp)
    (Evaluates.lookup rfl) (by simp [Mutable.sequence, found])
  have scanned := Domain.words_result words words heap address 0 found rfl entry
    [.return (.constant (.bool true))] true (fun _ => BlockReturns.return_value Evaluates.constant)
  simpa only [Bool.and_true] using scanned

private theorem score_lower (characters : List Char)
    (accepted : ∀ c ∈ characters, upper c = true) (initial : Int) :
    initial + characters.length ≤ characters.foldl (fun total c => total + letter c) initial := by
  induction characters generalizing initial with
  | nil => simp
  | cons head tail ih =>
      have headBounds := of_decide_eq_true (accepted head (by simp))
      have headLower : 1 ≤ letter head := by dsimp [letter]; omega
      have next := ih (fun c member => accepted c (by simp [member])) (initial+letter head)
      simp only [List.foldl_cons, List.length_cons, Int.natCast_add, Int.natCast_one]
      omega

private theorem score_positive (word : String) (accepted : uppercase word = true) : 0 < score word := by
  change (!word.toList.isEmpty && word.toList.all upper) = true at accepted
  obtain ⟨nonempty, letters⟩ := Bool.and_eq_true_iff.mp accepted
  have notNil : word.toList ≠ [] := by intro empty; simp [empty] at nonempty
  have lower := score_lower word.toList (fun c member => List.all_eq_true.mp letters c member) 0
  have length := List.length_pos_iff.mpr notNil
  change (0 : Int) + word.toList.length ≤ score word at lower
  omega

namespace solve
private abbrev predicate : Expr := .call (.name "is_triangle") [.call (.name "word_value") [.name "w"]]

private theorem word_test (word : String) (accepted : uppercase word = true)
    (heap : Mutable.Heap) (env : HeapCalls.Env) (getW : env "w" = some (string word)) :
    Evaluates program env predicate heap (flag (triangularTest (score word))) heap := by
  have positive := score_positive word accepted
  have letters : word.toList.all upper = true := (Bool.and_eq_true_iff.mp accepted).2
  have tested : ReturnsValue program "is_triangle" [number (score word)] heap
      (flag (triangularTest (score word))) heap := by
    simpa only [Int.toNat_of_nonneg (le_of_lt positive), triangularTest,
      if_neg (by omega : ¬ score word < 0)] using
      is_triangle.result (score word).toNat (Int.pos_iff_toNat_pos.mp positive) heap
  exact Evaluates.call
    (.cons (Evaluates.call (.cons (Evaluates.lookup getW) .nil) (word_value.result word letters heap)) .nil) tested

/-- Both generators count the same words without changing their shared input heap. -/
theorem result (words : List String) (heap : Mutable.Heap) (address : Nat)
    (found : heap[address]? = some (.list (words.map string))) (accepted : words.all uppercase = true) :
    ReturnsValue program "solve" [.ref address] heap (number (count words)) heap := by
  let items := words.attach
  have folded : items.foldl (fun total w => if triangularTest (score w.val) then total+1 else total) 0 =
      count words := List.foldl_attach (l := words)
        (f := fun total w => if triangularTest (score w) then total+1 else total)
  have sequence : Mutable.sequence (.ref address) heap =
      some (.ok (items.map (fun w => string w.val)), heap) := by
    simp [Mutable.sequence, items, found]
  apply BlockReturns.function rfl rfl
  apply BlockReturns.return_value
  rw [← folded]
  apply Evaluates.sum_generator_sequence (items := items.map (fun w => string w.val))
    (by intros; simp) (Evaluates.lookup rfl) sequence Evaluates.constant rfl
  apply ProjectEulerSolutions.FilteredSum.loop (source := .ref address) (heap := heap)
    (values := items) (encode := fun w => string w.val)
    (predicate := fun w => triangularTest (score w.val)) (number := fun _ => 1) items (by decide) sequence
  · intro env word getW
    exact word_test word.val (List.all_eq_true.mp accepted word.val word.property) heap env getW
  · intros
    exact Evaluates.constant
  · rfl
  · simp [setLocal]
end solve

private abbrev inputEnv (address : Nat) : HeapCalls.Env :=
  fun name => if name = "words" then some (.ref address) else none

private theorem condition_result (words : List String) (heap : Mutable.Heap) (address : Nat)
    (found : heap[address]? = some (.list (words.map string))) :
    Evaluates program (inputEnv address) Solvers.P42Ast.requires_solve_0 heap
      (flag (words.all uppercase)) heap :=
  Evaluates.call (.cons (Evaluates.lookup rfl) .nil) (words_are_uppercase.result words heap address found)

theorem requires_solve_0_supported (heap : Mutable.Heap) (address : Nat) (_valid : heap.Valid)
    (typed : Mutable.Value.HasType heap (.ref address) "list[str]") :
    (HeapCalls.evaluate program (inputEnv address) Solvers.P42Ast.requires_solve_0 heap).Supported := by
  obtain ⟨words, found⟩ := Mutable.Value.strings_of_hasType typed
  exact Execution.supported_of_finishes (condition_result words heap address found).finishes

theorem solve.«Equiv(naive)» (heap : Mutable.Heap) (address : Nat) (valid : heap.Valid)
    (typed : Mutable.Value.HasType heap (.ref address) "list[str]")
    (requires : HeapCalls.Condition program (inputEnv address) Solvers.P42Ast.requires_solve_0 heap) :
    (HeapCalls.call program "solve" [.ref address] heap).Equiv
      (HeapCalls.call P42Ast.module "naive" [.ref address] heap) := by
  obtain ⟨words, found⟩ := Mutable.Value.strings_of_hasType typed
  have accepted : words.all uppercase = true := by
    simpa using (condition_result words heap address found).condition_iff.mp requires
  have optimized := solve.result words heap address found accepted
  have reference := data.«project-euler-statements».data.py.project_euler_statements.p42.naive.result words heap address found
  exact Mutable.Call.equiv_of_finishes optimized.finishes reference.finishes
    (by simpa only [List.append_nil] using
      Mutable.Completion.equivalent_allocations heap [] [] (number (count words)) [] valid trivial (by simp))

theorem solve.«Supported» :
    (∀ (heap : Mutable.Heap) (address : Nat), heap.Valid → Mutable.Value.HasType heap (.ref address) "list[str]" →
      HeapCalls.Condition program (inputEnv address) Solvers.P42Ast.requires_solve_0 heap →
      (HeapCalls.call program "solve" [.ref address] heap).Supported) ∧
    (∀ (heap : Mutable.Heap) (address : Nat), heap.Valid → Mutable.Value.HasType heap (.ref address) "list[str]" →
      HeapCalls.Condition program (inputEnv address) Solvers.P42Ast.requires_solve_0 heap →
      (HeapCalls.call P42Ast.module "naive" [.ref address] heap).Supported) := by
  constructor
  · intro heap address _ typed requires
    obtain ⟨words, found⟩ := Mutable.Value.strings_of_hasType typed
    have accepted : words.all uppercase = true := by
      simpa using (condition_result words heap address found).condition_iff.mp requires
    exact (solve.result words heap address found accepted).returns.supported
  · intro heap address valid typed _
    exact data.«project-euler-statements».data.py.project_euler_statements.p42.naive.«Supported» heap address valid typed

end solvers.«42»
