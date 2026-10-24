import «.formalsw».solvers.«22.py.ast»
import data.«project-euler-statements».data.py.project_euler_statements.«p22.py»
import FormalSW

namespace solvers.«22»
open FormalSW.PyAst FormalSW.PyAst.HeapCalls
private abbrev program := Solvers.P22Ast.module
private abbrev number (n : Int) : Mutable.Value := .scalar (.int n)

/-- The domain used by the source prerequisite. -/
def upper (c : Char) : Bool := decide (65 ≤ c.toNat ∧ c.toNat ≤ 90)
def uppercase (name : String) : Bool := name.toList.all upper

/-- On the documented domain, the optimized unguarded ordinal sum equals the statement. -/
theorem name_value.result (name : String) (accepted : uppercase name = true) (heap : Mutable.Heap) :
    ReturnsValue program "name_value" [.scalar (.string name)] heap
      (number (ProjectEulerStatements.P22.nameValue name)) heap := by
  let characters := name.toList
  have total : characters.attach.foldl
      (fun (total : Int) c => total + ProjectEulerStatements.P22.letterValue c.val) 0 =
      (ProjectEulerStatements.P22.nameValue name : Int) := by
    rw [List.foldl_attach (f := fun (total : Int) (c : Char) => total + ProjectEulerStatements.P22.letterValue c)]
    exact data.«project-euler-statements».data.py.project_euler_statements.p22.fold_letter_values characters 0
  apply BlockReturns.function rfl rfl
  apply BlockReturns.return_value
  rw [← total]
  apply Evaluates.sum_generator_sequence
    (items := characters.map (fun c => .scalar (.string (String.singleton c)))) (by intros; simp)
    (Evaluates.lookup rfl) (by simp [Mutable.sequence, characters]) Evaluates.constant rfl
  apply sum_map_loop (values := characters.attach)
    (encode := fun c => .scalar (.string (String.singleton c.val)))
    (number := fun c => (ProjectEulerStatements.P22.letterValue c.val : Int)) characters.attach (by decide)
  · simp [sequenceValue, sequenceSource, Mutable.sequence, characters]
  · intro env c lookup
    have bounds : 65 ≤ c.val.toNat ∧ c.val.toNat ≤ 90 := by
      have good := List.all_eq_true.mp accepted c.val c.property
      simpa [upper] using good
    apply Evaluates.binary (av := number ((c.val.toNat : Int) - 65)) (bv := number 1)
      (middle := heap) (afterRight := heap)
    · apply Evaluates.binary (av := number c.val.toNat) (bv := number 65) (middle := heap) (afterRight := heap)
      · apply Evaluates.builtin (.cons (Evaluates.lookup lookup) .nil)
        change scalarResult (some (Strings.ord (.string (String.ofList [c.val])))) heap = _
        rw [Strings.ord_character]
        rfl
      · exact Evaluates.builtin (.cons Evaluates.constant .nil) rfl
      · rfl
    · exact Evaluates.constant
    · simp [Mutable.binary, scalarBinary, ProjectEulerStatements.P22.letterValue,
        show 'A'.toNat = 65 from rfl, show 'Z'.toNat = 90 from rfl, bounds, Int.natCast_sub bounds.1]
  · rfl
  · simp [setLocal]

namespace Domain
private def test : Expr := .unaryOp .not
  (.compare (.constant (.int 65)) [(.ltE, .builtinCall .ord [.name "c"]), (.ltE, .constant (.int 90))])
private def characterBody : List Stmt := [.ifThen test [.return (.constant (.bool false))] []]
private def nameBody : List Stmt := [.forIn (.name "c") (.name "name") characterBody []]

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
        omega
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

private theorem names_result (names remaining : List String) (heap : Mutable.Heap) (address position : Nat)
    (found : heap[address]? = some (.list (names.map (fun s => .scalar (.string s)))))
    (suffix : names.drop position = remaining) (env : Mutable.Env) (following : List Stmt) (flag : Bool)
    (rest : ∀ env, BlockReturns program env following heap (.scalar (.bool flag)) heap) :
    BlockReturns program env (.sequenceStepLoop "name" (.ref address) position nameBody [] :: following) heap
      (.scalar (.bool (remaining.all uppercase && flag))) heap := by
  have sequence : Mutable.sequence (.ref address) heap =
      some (.ok (names.map (fun s => .scalar (.string s))), heap) := by
    simp [Mutable.sequence, found]
  induction remaining generalizing position env with
  | nil =>
      apply BlockReturns.sequence_done (source := .ref address) (heap := heap) sequence
      · have missing : names[position]? = none := by
          simpa [suffix] using (List.head?_drop (l := names) (i := position)).symm
        simp [missing]
      · exact rest env
  | cons head tail ih =>
      have item : names[position]? = some head := by
        simpa [suffix] using (List.head?_drop (l := names) (i := position)).symm
      apply BlockReturns.sequence_step (source := .ref address) (heap := heap)
        (value := .scalar (.string head)) sequence (by simp [item])
      apply BlockReturns.for_sequence (value := .scalar (.string head)) (middle := heap)
        (items := head.toList.map (fun c => .scalar (.string (String.singleton c)))) (by intro expressions impossible; cases impossible)
        (Evaluates.lookup rfl) (by simp [Mutable.sequence])
      have next := fun env => ih (position + 1) (by simp [List.drop_add_one_eq_tail_drop, suffix]) env
      simpa [Calls.loopContinuation, Calls.loopControl, Calls.loopControlStmt, nameBody, characterBody, List.all_cons, uppercase, Bool.and_assoc] using
        characters_result head head.toList heap 0 rfl (setLocal env "name" (.scalar (.string head)))
          (.sequenceStepLoop "name" (.ref address) (position + 1) nameBody [] :: following)
          (tail.all uppercase && flag) next

end Domain

/-- The prerequisite checks every character and always terminates on a typed name list. -/
theorem names_are_uppercase.result (names : List String) (heap : Mutable.Heap) (address : Nat)
    (found : heap[address]? = some (.list (names.map (fun s => .scalar (.string s))))) :
    ReturnsValue program "names_are_uppercase" [.ref address] heap (.scalar (.bool (names.all uppercase))) heap := by
  let definition := (findFunctionIn program.body "names_are_uppercase").getD ([], [])
  let entry := functionEnv definition.1 definition.2 [.ref address] []
  apply BlockReturns.function (args := definition.1) (body := definition.2) rfl rfl
  apply BlockReturns.for_sequence (items := names.map (fun s => .scalar (.string s))) (middle := heap) (by intro expressions impossible; cases impossible)
    (Evaluates.lookup rfl) (by simp [Mutable.sequence, found])
  have scanned := Domain.names_result names names heap address 0 found rfl entry
    [.return (.constant (.bool true))] true (fun _ => BlockReturns.return_value Evaluates.constant)
  simpa only [Bool.and_true] using scanned

namespace Scoring
private def body : List Stmt := [.augAssign (.name "total") .add
  (.binOp (.name "i") .mult (.call (.name "name_value") [.name "nm"]))]

private theorem loop_result (names remaining : List String) (heap : Mutable.Heap) (address position : Nat)
    (found : heap[address]? = some (.list (names.map (fun s => .scalar (.string s)))))
    (accepted : ∀ name ∈ names, uppercase name = true) (suffix : names.drop position = remaining)
    (counter total : Int) (env : Mutable.Env) (lookup : env "total" = some (number total)) :
    ∃ finalEnv, BlockContinues program env [.enumerateStepLoop "i" "nm" (.ref address) position counter body []]
      heap finalEnv heap ∧ finalEnv "total" = some (number
        (indexedSum (fun i name => i * ProjectEulerStatements.P22.nameValue name) counter total remaining)) := by
  have sequence : Mutable.sequence (.ref address) heap =
      some (.ok (names.map (fun s => .scalar (.string s))), heap) := by
    simp [Mutable.sequence, found]
  induction remaining generalizing position counter total env with
  | nil =>
      refine ⟨env, ?_, lookup⟩
      apply BlockContinues.enumerate_done (source := .ref address) (heap := heap) sequence
      · have missing : names[position]? = none := by
          simpa [suffix] using (List.head?_drop (l := names) (i := position)).symm
        simp [missing]
      · exact BlockContinues.nil
  | cons head tail ih =>
      have item : names[position]? = some head := by
        simpa [suffix] using (List.head?_drop (l := names) (i := position)).symm
      have good : uppercase head = true := accepted head
        (List.mem_of_mem_drop (by rw [suffix]; simp))
      let boundEnv := setLocal (setLocal env "i" (number counter)) "nm" (.scalar (.string head))
      let nextEnv := setLocal boundEnv "total" (number (total + counter * ProjectEulerStatements.P22.nameValue head))
      have passed : BlockContinues program boundEnv body heap nextEnv heap := by
        apply BlockContinues.augAssign (right := number (counter * ProjectEulerStatements.P22.nameValue head))
          (middle := heap) (finalHeap := heap) (by simpa [boundEnv, setLocal] using lookup)
        · exact Evaluates.binary (Evaluates.lookup rfl)
            (Evaluates.call (.cons (Evaluates.lookup rfl) .nil) (name_value.result head good heap)) rfl
        · rfl
      obtain ⟨finalEnv, recurse, result⟩ := ih (position + 1)
        (by simp [List.drop_add_one_eq_tail_drop, suffix]) (counter + 1)
        (total + counter * ProjectEulerStatements.P22.nameValue head) nextEnv (by simp [nextEnv, setLocal])
      refine ⟨finalEnv, ?_, result⟩
      apply BlockContinues.enumerate_step (source := .ref address) (heap := heap)
        (value := .scalar (.string head)) sequence (by simp [item])
      apply BlockContinues.append passed
      simpa [Calls.loopContinuation, body] using recurse

private def definition : List Arg × List Stmt :=
  (findFunctionIn program.body "solve").getD ([], [])

/-- The optimized loop returns the independently proved reference score and preserves the input heap. -/
theorem result (names : List String) (heap : Mutable.Heap) (address : Nat)
    (found : heap[address]? = some (.list (names.map (fun s => .scalar (.string s)))))
    (accepted : names.all uppercase = true) :
    ReturnsValue program "solve" [.ref address] heap
      (number (data.«project-euler-statements».data.py.project_euler_statements.p22.scores names))
      (heap ++ [.list ((ScalarSort.strings names).map (fun s => .scalar (.string s)))]) := by
  let ordered := ScalarSort.strings names
  let allocated := heap ++ [.list (ordered.map (fun s => .scalar (.string s)))]
  let initial := setLocal (setLocal (functionEnv definition.1 definition.2 [.ref address] [])
    "names_sorted" (.ref heap.length)) "total" (number 0)
  have sorted : primitive (.builtin .sorted) [.ref address] heap = some (.ok (.ref heap.length), allocated) := by
    have operation := primitive_sorted_list heap address (names.map Constant.string) (ordered.map Constant.string)
      (by simpa only [List.map_map] using found) (ScalarSort.sort_strings names)
    simpa only [List.map_map] using operation
  have finalFound : allocated[heap.length]? = some (.list (ordered.map (fun s => .scalar (.string s)))) := by
    simp [allocated]
  have orderedAccepted : ∀ name ∈ ordered, uppercase name = true := by
    intro name member
    exact List.all_eq_true.mp accepted name ((ScalarSort.strings_perm names).mem_iff.mp member)
  obtain ⟨finalEnv, loop, result⟩ := loop_result ordered ordered allocated heap.length 0 finalFound orderedAccepted
    rfl 1 0 initial rfl
  apply BlockReturns.function (args := definition.1) (body := definition.2) rfl rfl
  apply BlockReturns.assign (value := .ref heap.length) (middle := allocated)
  · exact Evaluates.builtin (.cons (Evaluates.lookup rfl) .nil) sorted
  · apply BlockReturns.assign Evaluates.constant
    apply BlockContinues.returns (body := [.forIn (.tuple [.name "i", .name "nm"])
      (.call (.name "enumerate") [.name "names_sorted", .constant (.int 1)]) body []])
      (middleEnv := finalEnv) (middleHeap := allocated)
    · exact BlockContinues.for_enumerate_start (items := ordered.map (fun s => .scalar (.string s)))
        (Evaluates.lookup rfl) Evaluates.constant rfl
        (by simp [Mutable.sequence, finalFound]) loop
    · exact BlockReturns.return_value (Evaluates.lookup result)

end Scoring

private abbrev inputEnv (address : Nat) : Mutable.Env :=
  fun name => if name = "names" then some (.ref address) else none

private theorem condition_result (names : List String) (heap : Mutable.Heap) (address : Nat)
    (found : heap[address]? = some (.list (names.map (fun s => .scalar (.string s))))) :
    Evaluates program (inputEnv address) Solvers.P22Ast.requires_solve_0 heap
      (.scalar (.bool (names.all uppercase))) heap :=
  Evaluates.call (.cons (Evaluates.lookup rfl) .nil) (names_are_uppercase.result names heap address found)

theorem requires_solve_0_supported (heap : Mutable.Heap) (address : Nat) (_valid : heap.Valid)
    (typed : Mutable.Value.HasType heap (.ref address) "list[str]") :
    (HeapCalls.evaluate program (inputEnv address) Solvers.P22Ast.requires_solve_0 heap).Supported := by
  obtain ⟨names, found⟩ := Mutable.Value.strings_of_hasType typed
  exact Execution.supported_of_finishes (condition_result names heap address found).finishes

theorem solve.«Equiv(naive)» (heap : Mutable.Heap) (address : Nat) (valid : heap.Valid)
    (typed : Mutable.Value.HasType heap (.ref address) "list[str]")
    (requires : HeapCalls.Condition program (inputEnv address) Solvers.P22Ast.requires_solve_0 heap) :
    (HeapCalls.call program "solve" [.ref address] heap).Equiv
      (HeapCalls.call P22Ast.module "naive" [.ref address] heap) := by
  obtain ⟨names, found⟩ := Mutable.Value.strings_of_hasType typed
  have accepted : names.all uppercase = true := by
    simpa using (condition_result names heap address found).condition_iff.mp requires
  have optimized := Scoring.result names heap address found accepted
  have reference := data.«project-euler-statements».data.py.project_euler_statements.p22.naive.result names heap address found
  exact Mutable.Call.equiv_of_finishes optimized.finishes reference.finishes
    (Mutable.Completion.equivalent_allocations heap _ _
      (number (data.«project-euler-statements».data.py.project_euler_statements.p22.scores names)) [] valid trivial (by simp))

theorem solve.«Supported» :
    (∀ (heap : Mutable.Heap) (address : Nat), heap.Valid → Mutable.Value.HasType heap (.ref address) "list[str]" →
      HeapCalls.Condition program (inputEnv address) Solvers.P22Ast.requires_solve_0 heap →
      (HeapCalls.call program "solve" [.ref address] heap).Supported) ∧
    (∀ (heap : Mutable.Heap) (address : Nat), heap.Valid → Mutable.Value.HasType heap (.ref address) "list[str]" →
      HeapCalls.Condition program (inputEnv address) Solvers.P22Ast.requires_solve_0 heap →
      (HeapCalls.call P22Ast.module "naive" [.ref address] heap).Supported) := by
  constructor
  · intro heap address _ typed requires
    obtain ⟨names, found⟩ := Mutable.Value.strings_of_hasType typed
    have accepted : names.all uppercase = true := by
      simpa using (condition_result names heap address found).condition_iff.mp requires
    exact (Scoring.result names heap address found accepted).returns.supported
  · intro heap address valid typed _
    exact data.«project-euler-statements».data.py.project_euler_statements.p22.naive.«Supported» heap address valid typed

end solvers.«22»
