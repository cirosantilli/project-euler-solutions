import solvers.«22.py.lean.d».ScanRuntime
import lang.py.FormalSW.HeapRuntimeCostStringSorting
import lang.py.FormalSW.HeapRuntimeCostSetup
import Mathlib.Tactic.Linarith

namespace solvers.«22».runtimeCost
open FormalSW.PyAst HeapCalls HeapCalls.Runtime

abbrev program := Solvers.P22Ast.module
abbrev definition := Scan.definition
abbrev number := Name.number

theorem function_body_eq : executableBody definition.2 =
    [.assign [.name "names_sorted"] (.builtinCall .sorted [.name "names"]),
     .assign [.name "total"] (.constant (.int 0)),
     .forIn (.tuple [.name "i", .name "nm"])
       (.call (.name "enumerate") [.name "names_sorted", .constant (.int 1)]) Scan.body [],
     .return (.name "total")] := rfl

/-- Every name may have an arbitrary length. The uppercase prerequisite is
unnecessary for this runtime bound; it is used only by the equivalence proof. -/
theorem source_bound (names : List String) :
    TimeBound .word program "solve" [.ref 0] (StringList.heap names)
      (100 * (StringList.size names + 1)^2 * ((StringList.size names + 1).log2 + 1)) := by
  let entry := functionEnv definition.1 definition.2 [.ref 0] []
  let ordered := ScalarSort.strings names
  let allocated := StringList.heap names ++ [.list (StringList.encode ordered)]
  let withNames := setLocal entry "names_sorted" (.ref 1)
  let initial := setLocal withNames "total" (number 0)
  have sorted := EvaluatesIn.sorted_strings (arithmetic := .word) (module := program)
    (env := entry) (expression := .name "names") (heap := StringList.heap names)
    (EvaluatesIn.lookup rfl) (show (StringList.heap names)[0]? = some (.list (StringList.encode names)) from rfl)
  have initialized : BlockContinuesIn .word program withNames
      [.assign [.name "total"] (.constant (.int 0))] allocated 3 initial allocated :=
    BlockContinuesIn.assign EvaluatesIn.constant
  have found : allocated[1]? = some (.list (StringList.encode ordered)) := rfl
  obtain ⟨finalTotal, finalEnv, finalSaved, loop⟩ :=
    Scan.loop_cost ordered ordered 0 1 1 0 allocated initial found rfl (by simp [initial, setLocal])
  have enumerated := BlockContinuesIn.for_list_enumerate_start
    (EvaluatesIn.lookup (env := initial) (name := "names_sorted") (show initial "names_sorted" = some (.ref 1) from rfl))
    (EvaluatesIn.constant (value := .int 1)) found loop
  have returned : BlockReturnsIn .word program finalEnv [.return (.name "total")]
      allocated 2 (number finalTotal) allocated :=
    BlockReturnsIn.return_value (EvaluatesIn.lookup finalSaved)
  have body := (BlockContinuesIn.assign (name := "names_sorted") sorted).returns
    (initialized.returns (enumerated.returns returned))
  have called := BlockReturnsIn.function (module := program) (name := "solve")
    (args := definition.1) (body := definition.2) (values := [.ref 0]) rfl rfl
    (by simpa only [function_body_eq, List.cons_append, List.nil_append] using body)
  apply called.timeBound_append.mono
  change 3 + ((1 + SortWork.sorting .word (names.map Constant.string) + names.length + 3 + 2) +
    (3 + ((1 + 1 + ((ordered.map (fun text => 31 * text.length + 42)).sum + 7) + 7) + 2))) + 1 ≤ _
  have sorting := StringList.sorting_bound .word names
  have count := StringList.length_le_size names
  have scoring := StringList.weighted_sum_le_size 31 42 ordered
  have preserved : StringList.size ordered = StringList.size names :=
    StringList.size_perm (ScalarSort.strings_perm names)
  rw [preserved] at scoring
  change _ ≤ 42 * StringList.size names at scoring
  nlinarith [Nat.zero_le ((StringList.size names + 1).log2)]

end solvers.«22».runtimeCost

open FormalSW.PyAst.HeapCalls.Runtime in
theorem solvers.«22».solve.«BigO(n^2 * (Nat.log2 (n + 1) + 1), n = len(names) + sum(len(name) for name in names))» :
    FormalSW.PyAst.HeapCalls.Runtime.IsBigO .word Solvers.P22Ast.module "solve"
  (fun _ : List String => [.ref 0]) (fun input : List String => [.list (input.map (fun value => .scalar (.string value)))])
  (fun (input : List String) => (FormalSW.PyAst.HeapCalls.Condition Solvers.P22Ast.module (fun _arg => if _arg = "names" then some (.ref (0 : Nat)) else none) Solvers.P22Ast.requires_solve_0 [.list (input.map (fun value => .scalar (.string value)))])) (fun input : List String => FormalSW.PyAst.HeapCalls.Runtime.StringList.size input) (fun n : Nat => n^2 * (Nat.log2 (n + 1) + 1)) := by
  refine ⟨400, 1, fun names _ large => ?_⟩
  change 1 ≤ StringList.size names at large
  apply (solvers.«22».runtimeCost.source_bound names).mono
  have doubled : StringList.size names + 1 ≤ 2 * StringList.size names := by omega
  have squared := Nat.pow_le_pow_left doubled 2
  have scaled := Nat.mul_le_mul_left 100 squared
  have bound := Nat.mul_le_mul_right ((StringList.size names + 1).log2 + 1) scaled
  simpa only [Nat.mul_pow, ← Nat.mul_assoc] using bound
