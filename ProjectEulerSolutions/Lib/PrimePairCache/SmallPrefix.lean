import ProjectEulerSolutions.Lib.PrimePairCache.SmallRows000
import ProjectEulerSolutions.Lib.PrimePairCache.SmallRows001
import ProjectEulerSolutions.Lib.PrimePairCache.SmallRows002
import ProjectEulerSolutions.Lib.PrimePairCache.SmallRows003

namespace ProjectEulerSolutions.PrimePairCache
open PrimePairCertificate

def smallEmpty : Graph := .empty
theorem smallEmpty_agrees : Graph.Agrees PrimePairVertices.smallTree smallEmpty := Graph.empty_agrees _

def smallNode3 : Graph := .node 3 smallRow3 smallEmpty smallEmpty
theorem smallNode3_agrees : Graph.Agrees PrimePairVertices.smallTree smallNode3 :=
  Graph.node_agrees _ 3 smallRow3 smallEmpty smallEmpty
    smallEmpty_agrees smallEmpty_agrees (by decide) (by decide) smallRow3_checked

def smallNode7 : Graph := .node 7 smallRow7 smallNode3 smallEmpty
theorem smallNode7_agrees : Graph.Agrees PrimePairVertices.smallTree smallNode7 :=
  Graph.node_agrees _ 7 smallRow7 smallNode3 smallEmpty
    smallNode3_agrees smallEmpty_agrees (by decide) (by decide) smallRow7_checked

def smallNode13 : Graph := .node 13 smallRow13 smallEmpty smallEmpty
theorem smallNode13_agrees : Graph.Agrees PrimePairVertices.smallTree smallNode13 :=
  Graph.node_agrees _ 13 smallRow13 smallEmpty smallEmpty
    smallEmpty_agrees smallEmpty_agrees (by decide) (by decide) smallRow13_checked

def smallNode11 : Graph := .node 11 smallRow11 smallNode7 smallNode13
theorem smallNode11_agrees : Graph.Agrees PrimePairVertices.smallTree smallNode11 :=
  Graph.node_agrees _ 11 smallRow11 smallNode7 smallNode13
    smallNode7_agrees smallNode13_agrees (by decide) (by decide) smallRow11_checked

def smallNode19 : Graph := .node 19 smallRow19 smallEmpty smallEmpty
theorem smallNode19_agrees : Graph.Agrees PrimePairVertices.smallTree smallNode19 :=
  Graph.node_agrees _ 19 smallRow19 smallEmpty smallEmpty
    smallEmpty_agrees smallEmpty_agrees (by decide) (by decide) smallRow19_checked

def smallNode29 : Graph := .node 29 smallRow29 smallEmpty smallEmpty
theorem smallNode29_agrees : Graph.Agrees PrimePairVertices.smallTree smallNode29 :=
  Graph.node_agrees _ 29 smallRow29 smallEmpty smallEmpty
    smallEmpty_agrees smallEmpty_agrees (by decide) (by decide) smallRow29_checked

def smallNode23 : Graph := .node 23 smallRow23 smallNode19 smallNode29
theorem smallNode23_agrees : Graph.Agrees PrimePairVertices.smallTree smallNode23 :=
  Graph.node_agrees _ 23 smallRow23 smallNode19 smallNode29
    smallNode19_agrees smallNode29_agrees (by decide) (by decide) smallRow23_checked

def smallNode17 : Graph := .node 17 smallRow17 smallNode11 smallNode23
theorem smallNode17_agrees : Graph.Agrees PrimePairVertices.smallTree smallNode17 :=
  Graph.node_agrees _ 17 smallRow17 smallNode11 smallNode23
    smallNode11_agrees smallNode23_agrees (by decide) (by decide) smallRow17_checked

def smallNode37 : Graph := .node 37 smallRow37 smallEmpty smallEmpty
theorem smallNode37_agrees : Graph.Agrees PrimePairVertices.smallTree smallNode37 :=
  Graph.node_agrees _ 37 smallRow37 smallEmpty smallEmpty
    smallEmpty_agrees smallEmpty_agrees (by decide) (by decide) smallRow37_checked

def smallNode43 : Graph := .node 43 smallRow43 smallEmpty smallEmpty
theorem smallNode43_agrees : Graph.Agrees PrimePairVertices.smallTree smallNode43 :=
  Graph.node_agrees _ 43 smallRow43 smallEmpty smallEmpty
    smallEmpty_agrees smallEmpty_agrees (by decide) (by decide) smallRow43_checked

def smallNode41 : Graph := .node 41 smallRow41 smallNode37 smallNode43
theorem smallNode41_agrees : Graph.Agrees PrimePairVertices.smallTree smallNode41 :=
  Graph.node_agrees _ 41 smallRow41 smallNode37 smallNode43
    smallNode37_agrees smallNode43_agrees (by decide) (by decide) smallRow41_checked

def smallNode53 : Graph := .node 53 smallRow53 smallEmpty smallEmpty
theorem smallNode53_agrees : Graph.Agrees PrimePairVertices.smallTree smallNode53 :=
  Graph.node_agrees _ 53 smallRow53 smallEmpty smallEmpty
    smallEmpty_agrees smallEmpty_agrees (by decide) (by decide) smallRow53_checked

def smallNode61 : Graph := .node 61 smallRow61 smallEmpty smallEmpty
theorem smallNode61_agrees : Graph.Agrees PrimePairVertices.smallTree smallNode61 :=
  Graph.node_agrees _ 61 smallRow61 smallEmpty smallEmpty
    smallEmpty_agrees smallEmpty_agrees (by decide) (by decide) smallRow61_checked

def smallNode59 : Graph := .node 59 smallRow59 smallNode53 smallNode61
theorem smallNode59_agrees : Graph.Agrees PrimePairVertices.smallTree smallNode59 :=
  Graph.node_agrees _ 59 smallRow59 smallNode53 smallNode61
    smallNode53_agrees smallNode61_agrees (by decide) (by decide) smallRow59_checked

def smallNode47 : Graph := .node 47 smallRow47 smallNode41 smallNode59
theorem smallNode47_agrees : Graph.Agrees PrimePairVertices.smallTree smallNode47 :=
  Graph.node_agrees _ 47 smallRow47 smallNode41 smallNode59
    smallNode41_agrees smallNode59_agrees (by decide) (by decide) smallRow47_checked

def smallNode31 : Graph := .node 31 smallRow31 smallNode17 smallNode47
theorem smallNode31_agrees : Graph.Agrees PrimePairVertices.smallTree smallNode31 :=
  Graph.node_agrees _ 31 smallRow31 smallNode17 smallNode47
    smallNode17_agrees smallNode47_agrees (by decide) (by decide) smallRow31_checked

def smallNode71 : Graph := .node 71 smallRow71 smallEmpty smallEmpty
theorem smallNode71_agrees : Graph.Agrees PrimePairVertices.smallTree smallNode71 :=
  Graph.node_agrees _ 71 smallRow71 smallEmpty smallEmpty
    smallEmpty_agrees smallEmpty_agrees (by decide) (by decide) smallRow71_checked

def smallNode79 : Graph := .node 79 smallRow79 smallEmpty smallEmpty
theorem smallNode79_agrees : Graph.Agrees PrimePairVertices.smallTree smallNode79 :=
  Graph.node_agrees _ 79 smallRow79 smallEmpty smallEmpty
    smallEmpty_agrees smallEmpty_agrees (by decide) (by decide) smallRow79_checked

def smallNode73 : Graph := .node 73 smallRow73 smallNode71 smallNode79
theorem smallNode73_agrees : Graph.Agrees PrimePairVertices.smallTree smallNode73 :=
  Graph.node_agrees _ 73 smallRow73 smallNode71 smallNode79
    smallNode71_agrees smallNode79_agrees (by decide) (by decide) smallRow73_checked

def smallNode89 : Graph := .node 89 smallRow89 smallEmpty smallEmpty
theorem smallNode89_agrees : Graph.Agrees PrimePairVertices.smallTree smallNode89 :=
  Graph.node_agrees _ 89 smallRow89 smallEmpty smallEmpty
    smallEmpty_agrees smallEmpty_agrees (by decide) (by decide) smallRow89_checked

def smallNode101 : Graph := .node 101 smallRow101 smallEmpty smallEmpty
theorem smallNode101_agrees : Graph.Agrees PrimePairVertices.smallTree smallNode101 :=
  Graph.node_agrees _ 101 smallRow101 smallEmpty smallEmpty
    smallEmpty_agrees smallEmpty_agrees (by decide) (by decide) smallRow101_checked

def smallNode97 : Graph := .node 97 smallRow97 smallNode89 smallNode101
theorem smallNode97_agrees : Graph.Agrees PrimePairVertices.smallTree smallNode97 :=
  Graph.node_agrees _ 97 smallRow97 smallNode89 smallNode101
    smallNode89_agrees smallNode101_agrees (by decide) (by decide) smallRow97_checked

def smallNode83 : Graph := .node 83 smallRow83 smallNode73 smallNode97
theorem smallNode83_agrees : Graph.Agrees PrimePairVertices.smallTree smallNode83 :=
  Graph.node_agrees _ 83 smallRow83 smallNode73 smallNode97
    smallNode73_agrees smallNode97_agrees (by decide) (by decide) smallRow83_checked

def smallNode107 : Graph := .node 107 smallRow107 smallEmpty smallEmpty
theorem smallNode107_agrees : Graph.Agrees PrimePairVertices.smallTree smallNode107 :=
  Graph.node_agrees _ 107 smallRow107 smallEmpty smallEmpty
    smallEmpty_agrees smallEmpty_agrees (by decide) (by decide) smallRow107_checked

def smallNode113 : Graph := .node 113 smallRow113 smallEmpty smallEmpty
theorem smallNode113_agrees : Graph.Agrees PrimePairVertices.smallTree smallNode113 :=
  Graph.node_agrees _ 113 smallRow113 smallEmpty smallEmpty
    smallEmpty_agrees smallEmpty_agrees (by decide) (by decide) smallRow113_checked

def smallNode109 : Graph := .node 109 smallRow109 smallNode107 smallNode113
theorem smallNode109_agrees : Graph.Agrees PrimePairVertices.smallTree smallNode109 :=
  Graph.node_agrees _ 109 smallRow109 smallNode107 smallNode113
    smallNode107_agrees smallNode113_agrees (by decide) (by decide) smallRow109_checked

def smallNode131 : Graph := .node 131 smallRow131 smallEmpty smallEmpty
theorem smallNode131_agrees : Graph.Agrees PrimePairVertices.smallTree smallNode131 :=
  Graph.node_agrees _ 131 smallRow131 smallEmpty smallEmpty
    smallEmpty_agrees smallEmpty_agrees (by decide) (by decide) smallRow131_checked

def smallNode139 : Graph := .node 139 smallRow139 smallEmpty smallEmpty
theorem smallNode139_agrees : Graph.Agrees PrimePairVertices.smallTree smallNode139 :=
  Graph.node_agrees _ 139 smallRow139 smallEmpty smallEmpty
    smallEmpty_agrees smallEmpty_agrees (by decide) (by decide) smallRow139_checked

def smallNode137 : Graph := .node 137 smallRow137 smallNode131 smallNode139
theorem smallNode137_agrees : Graph.Agrees PrimePairVertices.smallTree smallNode137 :=
  Graph.node_agrees _ 137 smallRow137 smallNode131 smallNode139
    smallNode131_agrees smallNode139_agrees (by decide) (by decide) smallRow137_checked

def smallNode127 : Graph := .node 127 smallRow127 smallNode109 smallNode137
theorem smallNode127_agrees : Graph.Agrees PrimePairVertices.smallTree smallNode127 :=
  Graph.node_agrees _ 127 smallRow127 smallNode109 smallNode137
    smallNode109_agrees smallNode137_agrees (by decide) (by decide) smallRow127_checked

def smallNode103 : Graph := .node 103 smallRow103 smallNode83 smallNode127
theorem smallNode103_agrees : Graph.Agrees PrimePairVertices.smallTree smallNode103 :=
  Graph.node_agrees _ 103 smallRow103 smallNode83 smallNode127
    smallNode83_agrees smallNode127_agrees (by decide) (by decide) smallRow103_checked

def smallNode67 : Graph := .node 67 smallRow67 smallNode31 smallNode103
theorem smallNode67_agrees : Graph.Agrees PrimePairVertices.smallTree smallNode67 :=
  Graph.node_agrees _ 67 smallRow67 smallNode31 smallNode103
    smallNode31_agrees smallNode103_agrees (by decide) (by decide) smallRow67_checked

def smallPrefix : Graph := smallNode67
theorem smallPrefix_agrees : Graph.Agrees PrimePairVertices.smallTree smallPrefix := smallNode67_agrees
theorem smallPrefix_vertices : smallPrefix.vertices = PrimePairVertices.small.take 32 := rfl

theorem smallPrefix_vertices_subset (p : Nat) (member : p ∈ smallPrefix.vertices) : p ∈ PrimePairVertices.small := by
  rw [smallPrefix_vertices] at member
  exact List.mem_of_mem_take member

end ProjectEulerSolutions.PrimePairCache
