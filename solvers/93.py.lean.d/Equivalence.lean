import solvers.«93.py.lean.d».Search
import lang.py.FormalSW.MutableRelocation

namespace solvers.«93».solve
open FormalSW.PyAst
open Mutable.Relocation Graph.Relocation
open data.«project-euler-statements».data.py.project_euler_statements.p93.naive (answer)
set_option Elab.async false

private theorem tuple_equivalent (values : List Constant) (left right : Mutable.Heap) (a b : Nat)
    (first : left[a]? = some (.tuple (values.map Mutable.Value.scalar)))
    (second : right[b]? = some (.tuple (values.map Mutable.Value.scalar))) :
    Mutable.Completion.Equivalent ⟨.ok (.ref a), [], left, 0⟩ ⟨.ok (.ref b), [], right, 0⟩ := by
  let logical : Mutable.Completion := ⟨.ok (.ref 0), [], [.tuple (values.map Mutable.Value.scalar)], 0⟩
  have layout (heap : Mutable.Heap) (address : Nat)
      (found : heap[address]? = some (.tuple (values.map Mutable.Value.scalar))) :
      CompletionLayout logical ⟨.ok (.ref address), [], heap, 0⟩ (fun _ => address) := by
    refine ⟨⟨?_, ?_⟩, rfl, by simp [logical], rfl, rfl⟩
    · intro i hi j hj _
      simp only [logical, List.length_map, List.length_singleton] at hi hj
      omega
    · intro index node entry
      have bound := (List.getElem?_eq_some_iff.mp entry).1
      simp only [logical, List.length_map, List.length_singleton] at bound
      have zero : index = 0 := by omega
      subst index
      simp only [logical, List.map_cons, List.map_nil, List.getElem?_cons_zero, Option.some.injEq] at entry
      subst node
      simp [List.getElem?_map, found, relocateNode, graphNode, relocateEdge, Mutable.Value.edge,
        List.map_map, Function.comp_def]
  apply equivalent_completions logical _ _ (fun _ => a) (fun _ => b) (layout left a first) (layout right b second)
  · intro object member
    simp only [logical, List.mem_singleton] at member
    subst object
    intro value member
    obtain ⟨value, _, rfl⟩ := List.mem_map.mp member
    trivial
  · simp [logical]
  · simp [logical]
  · intro value outcome
    cases outcome
    simp [logical, Mutable.Value.edge, Graph.Edge.Within]

/-- Every possible optimized set schedule terminates. -/
theorem «Returns» (bound : Int) :
    (SetCalls.call program "solve" [.scalar (.int bound)] []).Returns := by
  intro choices
  obtain ⟨address, extra, finalUsed, executed, _⟩ := result bound [] [] choices 0 [] 0
  exact ⟨.ref address, [], extra, 0, executed.finishes (.done rfl)⟩

/-- All traversal schedules return the same digit tuple as the complete reference search. -/
theorem «Equiv(naive)» (bound : Int) :
    (SetCalls.call program "solve" [.scalar (.int bound)] []).Equiv
      (SetCalls.call P93Ast.module "naive" [.scalar (.int bound)] []) := by
  intro leftChoices rightChoices
  obtain ⟨leftAddress, leftExtra, _, executed, leftFound⟩ := result bound [] [] leftChoices 0 [] 0
  obtain ⟨rightAddress, rightExtra, reference, rightFound⟩ :=
    data.«project-euler-statements».data.py.project_euler_statements.p93.naive.result bound [] []
  apply Mutable.Call.equiv_of_finishes
    (a := (SetCalls.call program "solve" [.scalar (.int bound)] []).run leftChoices)
    (b := (SetCalls.call P93Ast.module "naive" [.scalar (.int bound)] []).run rightChoices)
    (executed.finishes (.done rfl)) (SetCalls.finishes_of_heap reference.finishes rightChoices 0)
  apply tuple_equivalent ((answer bound).map Constant.int)
  · simpa only [List.map_map] using leftFound
  · simpa only [List.map_map] using rightFound

theorem «Supported» :
    (∀ bound : Int, (SetCalls.call program "solve" [.scalar (.int bound)] []).Supported) ∧
    (∀ bound : Int, (SetCalls.call P93Ast.module "naive" [.scalar (.int bound)] []).Supported) := by
  constructor
  · intro bound choices
    exact («Returns» bound choices).supported
  · intro bound choices
    obtain ⟨_, _, returned, _⟩ :=
      data.«project-euler-statements».data.py.project_euler_statements.p93.naive.result bound [] []
    exact (SetCalls.returns_of_heap returned.returns choices).supported

theorem default_returns : (SetCalls.call program "solve" [] []).Returns := by
  intro choices
  obtain ⟨address, extra, _, executed, _⟩ := default_result [] [] choices 0 [] 0
  exact ⟨.ref address, [], extra, 0, executed.finishes (.done rfl)⟩

theorem default_equiv :
    (SetCalls.call program "solve" [] []).Equiv (SetCalls.call P93Ast.module "naive" [] []) := by
  intro leftChoices rightChoices
  obtain ⟨_, _, _, executed, leftFound⟩ := default_result [] [] leftChoices 0 [] 0
  obtain ⟨_, _, reference, rightFound⟩ :=
    data.«project-euler-statements».data.py.project_euler_statements.p93.naive.default_result [] []
  apply Mutable.Call.equiv_of_finishes
    (a := (SetCalls.call program "solve" [] []).run leftChoices)
    (b := (SetCalls.call P93Ast.module "naive" [] []).run rightChoices)
    (executed.finishes (.done rfl)) (SetCalls.finishes_of_heap reference.finishes rightChoices 0)
  apply tuple_equivalent ((answer 9).map Constant.int)
  · simpa only [List.map_map] using leftFound
  · simpa only [List.map_map] using rightFound

end solvers.«93».solve
