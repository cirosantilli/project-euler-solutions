import ProjectEulerSolutions.Lib.PrimePairLarge.Bounds

namespace ProjectEulerSolutions.PrimePairLarge
open PrimePairCertificate

def row9643_small : List Nat := []
theorem row9643_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 9643 row9643_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row9643_block_005 : List Nat := [9697]
def plan9643_block_005 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, (.firstFactor 19), .first, .first, (.firstFactor 7), .first, .possible, .first, (.firstFactor 7), (.firstFactor 13), (.firstFactor 11), .first, .first, .first, .second, .second, .second, .first, .first, (.firstFactor 13), .first, (.secondFactor 13), .first, .first, .first, .first, (.secondFactor 17), (.firstFactor 11), (.secondFactor 7), .first, .second, (.secondFactor 13), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 13), (.firstFactor 7), .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, (.secondFactor 11), .first, .first, (.secondFactor 7), (.secondFactor 31), .first, .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 17), .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 11), .first, (.firstFactor 31), .first, .first, (.firstFactor 11), .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11)]
set_option maxRecDepth 2048 in
theorem row9643_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 9643 row9643_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan9643_block_005
  · decide
  · rfl

def row9643_block_006 : List Nat := [11497]
def plan9643_block_006 : List Budget.Choice := [.first, (.firstFactor 13), .first, (.secondFactor 7), .first, (.secondFactor 13), .first, (.firstFactor 7), .first, (.firstFactor 13), .first, .first, (.firstFactor 7), (.secondFactor 7), .first, (.secondFactor 11), .first, (.firstFactor 11), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 29), (.secondFactor 19), (.secondFactor 31), (.secondFactor 11), .first, (.firstFactor 37), .first, .first, (.firstFactor 7), (.secondFactor 7), (.secondFactor 11), .first, .first, (.firstFactor 7), (.firstFactor 17), .first, (.firstFactor 13), .first, (.firstFactor 11), (.firstFactor 7), .first, .first, .first, (.secondFactor 11), (.firstFactor 13), .first, .first, (.secondFactor 7), (.firstFactor 23), (.firstFactor 37), .first, .first, .first, .second, .first, .first, (.firstFactor 29), (.secondFactor 11), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.secondFactor 17), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, (.secondFactor 29), .first, (.secondFactor 23), .first, .second, (.firstFactor 7), .first, .first, .first, (.secondFactor 17), (.firstFactor 7), (.secondFactor 7), (.firstFactor 23), .first, .first, .first, (.firstFactor 37), .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .possible, (.firstFactor 19), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 17), (.firstFactor 7), .first, (.firstFactor 19), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, (.secondFactor 13), .first, (.firstFactor 7), (.firstFactor 19), .first, .first, (.secondFactor 13), .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, .second, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 11), (.firstFactor 13), .first, (.firstFactor 7), .first, .first, (.firstFactor 23), .first, (.firstFactor 13), .first, .first, (.secondFactor 11), .first, (.firstFactor 19), (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, (.firstFactor 37), .first, .first, .first, .first, (.secondFactor 7), .first, .second, (.firstFactor 17), .first, .first, (.secondFactor 23), .first, .second, .first, (.firstFactor 29), (.firstFactor 7), .first, (.secondFactor 17), .first]
set_option maxRecDepth 2048 in
theorem row9643_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 9643 row9643_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan9643_block_006
  · decide
  · rfl

def row9643_block_007 : List Nat := [12721, 12829, 13177, 14401]
def plan9643_block_007 : List Budget.Choice := [(.firstFactor 23), (.firstFactor 7), .first, .second, .second, .first, .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 11), (.firstFactor 7), .first, .first, .first, (.firstFactor 11), (.firstFactor 29), .first, (.secondFactor 11), (.firstFactor 13), (.secondFactor 19), (.firstFactor 7), .first, (.firstFactor 11), (.secondFactor 23), (.secondFactor 13), .first, .first, .first, .first, .first, .first, (.firstFactor 23), (.secondFactor 19), .first, .possible, (.firstFactor 13), .first, (.firstFactor 19), (.secondFactor 29), (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 7), .possible, (.secondFactor 11), (.firstFactor 7), (.secondFactor 31), .first, .first, (.firstFactor 37), .first, .first, (.secondFactor 17), .first, .first, .first, .first, .second, (.firstFactor 13), (.firstFactor 7), .first, .first, .second, .first, (.firstFactor 11), (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 19), .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.secondFactor 11), .possible, (.secondFactor 13), .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 23), .first, .first, .second, .second, (.secondFactor 23), .first, (.firstFactor 19), .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 7), (.secondFactor 7), (.secondFactor 13), .first, (.firstFactor 7), .first, .first, .first, .first, .second, .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 17), (.secondFactor 11), .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 23), (.firstFactor 17), .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 31), .first, (.firstFactor 13), .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 11), (.secondFactor 11), .first, .first, (.firstFactor 11), .first, (.firstFactor 19), (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.firstFactor 13), .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, .first, (.secondFactor 7), (.firstFactor 7), (.firstFactor 29), .first, .first, (.firstFactor 7), .first, (.firstFactor 17), (.secondFactor 17), .first, .first, .first, .possible, (.firstFactor 7), .first, (.secondFactor 7), .first, (.secondFactor 13), .second, .first, (.firstFactor 7), (.firstFactor 11)]
set_option maxRecDepth 2048 in
theorem row9643_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 9643 row9643_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan9643_block_007
  · decide
  · rfl

def row9643_block_008 : List Nat := [14731, 15277, 15559, 15901, 16333, 16369]
def plan9643_block_008 : List Budget.Choice := [.first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 31), (.secondFactor 11), .first, (.secondFactor 31), .first, (.firstFactor 11), .first, .first, (.firstFactor 23), .first, .first, (.secondFactor 17), .first, .first, (.firstFactor 37), .first, (.secondFactor 7), .first, .first, .possible, (.firstFactor 31), .first, .first, .first, .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 11), (.firstFactor 7), .first, .first, (.firstFactor 17), .first, (.firstFactor 7), .first, (.secondFactor 11), .first, .first, (.firstFactor 11), .first, .first, .first, .first, .first, .first, (.secondFactor 19), .first, .first, .first, .first, (.secondFactor 17), .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, .second, .first, .first, .first, (.firstFactor 11), .second, (.secondFactor 29), (.secondFactor 7), .first, .first, (.secondFactor 23), (.firstFactor 17), .first, .first, .first, .possible, .first, (.firstFactor 7), .first, (.secondFactor 31), (.firstFactor 13), (.firstFactor 11), .first, (.firstFactor 7), (.firstFactor 37), .first, (.firstFactor 17), (.firstFactor 7), .first, .first, (.firstFactor 13), .first, .first, (.secondFactor 7), (.secondFactor 19), .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 29), .first, (.firstFactor 7), .first, .possible, .first, .first, (.firstFactor 7), (.secondFactor 13), (.firstFactor 19), .first, .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 23), .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 7), .first, .first, (.secondFactor 7), .possible, .first, (.firstFactor 11), (.firstFactor 7), .first, (.firstFactor 13), .first, .first, (.firstFactor 17), (.secondFactor 13), .first, .first, (.firstFactor 29), (.secondFactor 7), .first, (.firstFactor 19), .first, (.secondFactor 13), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 7), (.secondFactor 7), .first, .second, .first, .first, .first, .first, .first, (.firstFactor 13), .first, (.secondFactor 7), (.secondFactor 11), .first, .first, .possible, (.firstFactor 7), .first, .first, .first, .possible, (.firstFactor 7), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row9643_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 9643 row9643_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan9643_block_008
  · decide
  · rfl

def row9643_block_009 : List Nat := []
theorem row9643_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 9643 row9643_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row9643_block_010 : List Nat := []
theorem row9643_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 9643 row9643_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row9643_block_011 : List Nat := []
theorem row9643_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 9643 row9643_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row9643_block_012 : List Nat := []
theorem row9643_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 9643 row9643_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row9643_block_013 : List Nat := []
theorem row9643_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 9643 row9643_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row9643_large_left_left : List Nat := row9643_small ++ row9643_block_005
theorem row9643_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 9643 row9643_large_left_left :=
  Budget.RowUpperBound.branch row9643_small_covers row9643_block_005_covers

def row9643_large_left_right_right : List Nat := row9643_block_007 ++ row9643_block_008
theorem row9643_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 9643 row9643_large_left_right_right :=
  Budget.RowUpperBound.branch row9643_block_007_covers row9643_block_008_covers

def row9643_large_left_right : List Nat := row9643_block_006 ++ row9643_large_left_right_right
theorem row9643_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 9643 row9643_large_left_right :=
  Budget.RowUpperBound.branch row9643_block_006_covers row9643_large_left_right_right_covers

def row9643_large_left : List Nat := row9643_large_left_left ++ row9643_large_left_right
theorem row9643_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 9643 row9643_large_left :=
  Budget.RowUpperBound.branch row9643_large_left_left_covers row9643_large_left_right_covers

def row9643_large_right_left : List Nat := row9643_block_009 ++ row9643_block_010
theorem row9643_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 9643 row9643_large_right_left :=
  Budget.RowUpperBound.branch row9643_block_009_covers row9643_block_010_covers

def row9643_large_right_right_right : List Nat := row9643_block_012 ++ row9643_block_013
theorem row9643_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 9643 row9643_large_right_right_right :=
  Budget.RowUpperBound.branch row9643_block_012_covers row9643_block_013_covers

def row9643_large_right_right : List Nat := row9643_block_011 ++ row9643_large_right_right_right
theorem row9643_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 9643 row9643_large_right_right :=
  Budget.RowUpperBound.branch row9643_block_011_covers row9643_large_right_right_right_covers

def row9643_large_right : List Nat := row9643_large_right_left ++ row9643_large_right_right
theorem row9643_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 9643 row9643_large_right :=
  Budget.RowUpperBound.branch row9643_large_right_left_covers row9643_large_right_right_covers

def row9643_large : List Nat := row9643_large_left ++ row9643_large_right
theorem row9643_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 9643 row9643_large :=
  Budget.RowUpperBound.branch row9643_large_left_covers row9643_large_right_covers

def row9643 : List Nat := row9643_large
theorem row9643_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 9643 row9643 := row9643_large_covers

def row9649_small : List Nat := []
theorem row9649_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 9649 row9649_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row9649_block_005 : List Nat := [10273]
def plan9649_block_005 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, (.secondFactor 7), .first, (.secondFactor 17), .first, (.firstFactor 7), .first, .first, (.firstFactor 11), (.firstFactor 7), .first, .first, .first, (.firstFactor 17), (.firstFactor 7), (.secondFactor 7), .first, .first, (.firstFactor 37), .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 13), (.secondFactor 17), .first, .second, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, (.firstFactor 17), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.firstFactor 29), .first, .first, (.firstFactor 7), (.firstFactor 23), .first, (.firstFactor 11), .first, .first, (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, .first, .possible, .first, .first, (.firstFactor 7), .first, (.secondFactor 13), .first, (.secondFactor 7), .first, .first, (.secondFactor 23), .first]
set_option maxRecDepth 2048 in
theorem row9649_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 9649 row9649_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan9649_block_005
  · decide
  · rfl

def row9649_block_006 : List Nat := [11149, 11527, 11743, 12211, 12301]
def plan9649_block_006 : List Budget.Choice := [.first, (.secondFactor 13), .first, (.firstFactor 7), .first, (.firstFactor 13), .first, (.firstFactor 19), .first, (.secondFactor 13), .first, .first, (.secondFactor 7), (.firstFactor 7), .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 23), (.secondFactor 11), .first, .first, (.firstFactor 13), .first, .first, (.secondFactor 7), (.firstFactor 7), (.secondFactor 31), .first, .first, (.firstFactor 37), (.firstFactor 11), .first, (.firstFactor 23), .first, .second, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 31), .first, .first, (.firstFactor 7), (.firstFactor 11), (.secondFactor 23), .first, (.firstFactor 17), .first, .first, .first, .first, (.secondFactor 11), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .second, .second, .first, .first, (.firstFactor 29), .first, (.secondFactor 11), (.secondFactor 7), .possible, .first, .second, .first, (.secondFactor 7), .first, (.firstFactor 37), .first, (.firstFactor 31), .first, (.secondFactor 11), (.firstFactor 19), .first, .first, .first, (.firstFactor 29), (.firstFactor 11), (.firstFactor 7), (.secondFactor 11), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 13), (.firstFactor 11), .first, .first, .possible, .first, (.firstFactor 17), .first, .first, (.secondFactor 7), .first, (.firstFactor 23), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 13), .first, (.secondFactor 7), (.firstFactor 7), .possible, .first, (.firstFactor 13), .first, .first, .first, .first, .first, (.secondFactor 17), (.firstFactor 11), .first, .first, .second, (.firstFactor 37), .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 17), .first, .first, .first, (.firstFactor 7), .second, (.firstFactor 11), .first, (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 13), .first, (.secondFactor 11), .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 17), .first, .first, .possible, .first, .first, (.secondFactor 11), .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .possible, .first, .first, .first, .first, (.secondFactor 11), .first, (.secondFactor 19), (.secondFactor 7), .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row9649_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 9649 row9649_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan9649_block_006
  · decide
  · rfl

def row9649_block_007 : List Nat := [13003, 13171, 13681, 13933]
def plan9649_block_007 : List Budget.Choice := [(.firstFactor 11), (.secondFactor 7), .first, (.secondFactor 29), .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 19), (.secondFactor 7), .first, .first, (.firstFactor 31), .first, (.firstFactor 11), .first, .second, (.firstFactor 23), (.secondFactor 31), (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 11), (.firstFactor 13), .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 11), .first, (.firstFactor 23), (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 29), .first, .first, (.firstFactor 7), .second, (.secondFactor 17), (.firstFactor 19), .second, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 19), (.firstFactor 37), (.firstFactor 17), .first, .first, .possible, .first, .second, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .second, (.firstFactor 31), .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .possible, (.secondFactor 19), (.firstFactor 13), .first, .first, .first, .first, .first, (.secondFactor 17), .first, .first, (.secondFactor 19), (.secondFactor 11), (.firstFactor 19), .first, (.firstFactor 7), .first, .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 7), (.firstFactor 7), (.firstFactor 13), .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 11), .first, .first, .second, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 17), (.secondFactor 13), .first, .first, (.secondFactor 11), (.secondFactor 19), .first, (.secondFactor 23), .first, .possible, (.firstFactor 23), .first, (.firstFactor 17), .first, .first, .first, .first, (.firstFactor 29), (.firstFactor 13), .first, .first, (.secondFactor 11), .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, (.firstFactor 7), .first, (.firstFactor 19), .first, .first, (.secondFactor 7), .first, .first, .first, .first, .possible, (.firstFactor 13), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 31), .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, (.firstFactor 19), (.secondFactor 13), .second, .first, .first, .second, .first, (.firstFactor 13), .first, (.firstFactor 19), .first, .first, (.firstFactor 7), (.secondFactor 7), (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .second, .first, .first, .first, .second, (.firstFactor 11), (.firstFactor 17), .first, (.firstFactor 7), .first, (.firstFactor 13), .second, .first, (.firstFactor 19), (.firstFactor 7)]
set_option maxRecDepth 2048 in
theorem row9649_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 9649 row9649_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan9649_block_007
  · decide
  · rfl

def row9649_block_008 : List Nat := [14479, 14737, 15439, 16231]
def plan9649_block_008 : List Budget.Choice := [.possible, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 11), .first, .first, (.firstFactor 19), .first, (.firstFactor 29), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 23), .first, .first, (.secondFactor 11), .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .possible, .first, .first, .first, .first, (.firstFactor 29), .first, (.secondFactor 17), .first, (.firstFactor 7), .first, (.firstFactor 13), (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 19), .first, .first, (.firstFactor 7), (.firstFactor 11), .first, (.secondFactor 11), .first, .first, .first, (.secondFactor 17), (.secondFactor 11), .first, .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 17), .first, .first, .first, .first, .first, .first, (.secondFactor 17), (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 11), .first, (.firstFactor 13), .first, .first, (.secondFactor 13), .second, .first, (.firstFactor 31), .first, .first, (.firstFactor 19), (.secondFactor 7), .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .possible, .first, .second, .first, .first, .first, (.secondFactor 17), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 13), (.secondFactor 11), (.firstFactor 23), .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .second, .first, (.firstFactor 17), .first, (.secondFactor 11), .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 37), (.secondFactor 7), .first, .first, (.firstFactor 7), (.secondFactor 17), .second, (.firstFactor 13), (.firstFactor 11), .first, (.firstFactor 17), .first, .first, (.firstFactor 7), (.firstFactor 13), .first, .first, (.firstFactor 23), (.firstFactor 7), .first, .first, .first, (.firstFactor 13), .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 7), .first, .first, .first, .first, .first, .first, .possible, (.firstFactor 11), .first, (.firstFactor 7), (.firstFactor 19), .first, .first, (.firstFactor 29), (.secondFactor 7), .first, .first, .second, (.secondFactor 19), (.firstFactor 11), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row9649_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 9649 row9649_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan9649_block_008
  · decide
  · rfl

def row9649_block_009 : List Nat := []
theorem row9649_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 9649 row9649_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row9649_block_010 : List Nat := []
theorem row9649_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 9649 row9649_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row9649_block_011 : List Nat := []
theorem row9649_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 9649 row9649_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row9649_block_012 : List Nat := []
theorem row9649_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 9649 row9649_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row9649_block_013 : List Nat := []
theorem row9649_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 9649 row9649_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row9649_large_left_left : List Nat := row9649_small ++ row9649_block_005
theorem row9649_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 9649 row9649_large_left_left :=
  Budget.RowUpperBound.branch row9649_small_covers row9649_block_005_covers

def row9649_large_left_right_right : List Nat := row9649_block_007 ++ row9649_block_008
theorem row9649_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 9649 row9649_large_left_right_right :=
  Budget.RowUpperBound.branch row9649_block_007_covers row9649_block_008_covers

def row9649_large_left_right : List Nat := row9649_block_006 ++ row9649_large_left_right_right
theorem row9649_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 9649 row9649_large_left_right :=
  Budget.RowUpperBound.branch row9649_block_006_covers row9649_large_left_right_right_covers

def row9649_large_left : List Nat := row9649_large_left_left ++ row9649_large_left_right
theorem row9649_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 9649 row9649_large_left :=
  Budget.RowUpperBound.branch row9649_large_left_left_covers row9649_large_left_right_covers

def row9649_large_right_left : List Nat := row9649_block_009 ++ row9649_block_010
theorem row9649_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 9649 row9649_large_right_left :=
  Budget.RowUpperBound.branch row9649_block_009_covers row9649_block_010_covers

def row9649_large_right_right_right : List Nat := row9649_block_012 ++ row9649_block_013
theorem row9649_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 9649 row9649_large_right_right_right :=
  Budget.RowUpperBound.branch row9649_block_012_covers row9649_block_013_covers

def row9649_large_right_right : List Nat := row9649_block_011 ++ row9649_large_right_right_right
theorem row9649_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 9649 row9649_large_right_right :=
  Budget.RowUpperBound.branch row9649_block_011_covers row9649_large_right_right_right_covers

def row9649_large_right : List Nat := row9649_large_right_left ++ row9649_large_right_right
theorem row9649_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 9649 row9649_large_right :=
  Budget.RowUpperBound.branch row9649_large_right_left_covers row9649_large_right_right_covers

def row9649_large : List Nat := row9649_large_left ++ row9649_large_right
theorem row9649_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 9649 row9649_large :=
  Budget.RowUpperBound.branch row9649_large_left_covers row9649_large_right_covers

def row9649 : List Nat := row9649_large
theorem row9649_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 9649 row9649 := row9649_large_covers

def row9661_small : List Nat := []
theorem row9661_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 9661 row9661_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row9661_block_005 : List Nat := []
def plan9661_block_005 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, (.firstFactor 13), .first, (.secondFactor 23), .first, (.firstFactor 11), (.firstFactor 7), (.secondFactor 29), .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 7), (.secondFactor 17), .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 19), .first, (.firstFactor 7), (.secondFactor 19), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 29), (.secondFactor 7), .first, (.firstFactor 23), .first, .first, .first, .first, .second, .first, .first, (.firstFactor 13), (.secondFactor 7), .first, (.secondFactor 23), .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .second, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, .second, .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 17), (.firstFactor 7)]
set_option maxRecDepth 2048 in
theorem row9661_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 9661 row9661_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan9661_block_005
  · decide
  · rfl

def row9661_block_006 : List Nat := [12109]
def plan9661_block_006 : List Budget.Choice := [.first, .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 17), .first, (.secondFactor 7), .first, .first, (.secondFactor 31), (.secondFactor 11), .first, .first, .first, (.firstFactor 31), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 11), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.secondFactor 11), .first, (.secondFactor 7), .first, .first, (.firstFactor 31), (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 7), (.firstFactor 23), .first, .first, .first, .first, (.firstFactor 17), .first, .first, .first, .second, (.secondFactor 11), .first, (.firstFactor 31), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 19), .first, .first, (.firstFactor 11), .first, (.secondFactor 31), .first, (.firstFactor 17), (.firstFactor 7), .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 29), (.secondFactor 7), .first, (.secondFactor 19), .first, (.firstFactor 17), .first, .first, .first, (.secondFactor 11), .first, (.firstFactor 7), (.secondFactor 17), .first, .first, .first, .first, .first, (.firstFactor 11), (.secondFactor 7), .first, (.firstFactor 37), .first, (.secondFactor 23), .first, (.secondFactor 13), .first, .first, .first, .first, (.secondFactor 11), (.firstFactor 11), .first, .first, .first, .first, .first, (.secondFactor 23), (.firstFactor 13), (.firstFactor 7), .first, (.firstFactor 23), .first, (.firstFactor 37), .first, (.firstFactor 7), .second, .first, (.secondFactor 13), .first, .first, .first, (.firstFactor 19), .first, .first, .first, (.secondFactor 11), .first, .first, (.firstFactor 13), .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.secondFactor 7), (.secondFactor 29), .first, (.secondFactor 11), (.firstFactor 7), (.secondFactor 7), .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 23), .first, (.secondFactor 19), .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 13), (.firstFactor 7), .first, (.secondFactor 7), (.secondFactor 11), .first, .first, .possible, .first, .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 11), .first, .first, .first, .first, .first, (.firstFactor 31), .first, .first, .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 17), .first, .first, (.firstFactor 29), .first]
set_option maxRecDepth 2048 in
theorem row9661_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 9661 row9661_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan9661_block_006
  · decide
  · rfl

def row9661_block_007 : List Nat := [12541, 12781, 12853, 13513]
def plan9661_block_007 : List Budget.Choice := [.second, (.firstFactor 11), .first, (.secondFactor 7), (.secondFactor 23), .first, .first, .second, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .possible, (.secondFactor 31), (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 29), (.firstFactor 13), (.firstFactor 17), .first, (.firstFactor 31), (.secondFactor 7), (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 17), .first, (.firstFactor 7), .second, .first, (.firstFactor 29), (.firstFactor 7), .possible, .first, (.firstFactor 31), .first, .first, (.firstFactor 13), (.firstFactor 11), .first, .possible, (.firstFactor 7), .first, .first, (.firstFactor 17), .first, .first, (.secondFactor 31), .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 17), .second, .first, .first, .first, .second, (.firstFactor 11), (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 23), .first, (.firstFactor 11), .first, (.firstFactor 31), (.secondFactor 13), (.firstFactor 7), .first, .first, (.secondFactor 11), .first, .first, (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 11), (.secondFactor 17), (.firstFactor 7), .first, (.firstFactor 37), .first, .first, .first, .first, .first, .first, (.secondFactor 17), (.secondFactor 13), (.firstFactor 17), .first, (.secondFactor 19), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .possible, .first, .first, .first, (.secondFactor 13), .first, .first, (.firstFactor 23), .first, .first, (.firstFactor 29), .first, .first, (.secondFactor 7), .first, (.firstFactor 13), (.firstFactor 7), .first, .second, .first, .first, (.secondFactor 7), .first, (.firstFactor 17), (.firstFactor 7), .first, .first, (.firstFactor 13), .first, .first, .first, .first, (.secondFactor 17), .first, .first, .first, .first, (.firstFactor 23), .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 23), (.firstFactor 17), .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 11), .first, (.firstFactor 7), (.secondFactor 11), (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 31), .first, .first, (.secondFactor 23), (.firstFactor 11), .first, .first, .first, (.firstFactor 29), .first, (.secondFactor 7), (.firstFactor 11), .first, .first, (.secondFactor 23), (.firstFactor 7), (.secondFactor 11), .first, (.secondFactor 17), .first, .second, (.firstFactor 17), .first, .first, (.firstFactor 13)]
set_option maxRecDepth 2048 in
theorem row9661_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 9661 row9661_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan9661_block_007
  · decide
  · rfl

def row9661_block_008 : List Nat := [14683, 15919, 16033]
def plan9661_block_008 : List Budget.Choice := [(.firstFactor 11), .first, (.secondFactor 13), .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 23), .first, (.secondFactor 7), .first, .first, .second, .first, .first, (.firstFactor 7), .first, .first, .possible, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .second, .first, (.firstFactor 7), (.secondFactor 17), .first, .first, (.firstFactor 13), .first, (.firstFactor 19), .first, (.secondFactor 7), .first, .first, (.secondFactor 19), (.firstFactor 13), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 19), (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .second, .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 29), (.firstFactor 7), (.firstFactor 31), .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 13), .first, (.secondFactor 11), (.secondFactor 7), .first, (.secondFactor 13), .first, .first, .first, (.firstFactor 23), .first, .first, .second, (.firstFactor 19), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 29), .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 11), (.secondFactor 19), (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .second, (.firstFactor 7), (.firstFactor 11), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 11), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.secondFactor 11), .second, .first, .first, (.firstFactor 29), .second, (.secondFactor 13), (.firstFactor 7), .possible, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 37), .first, .first, .possible, (.secondFactor 11), .first, (.firstFactor 11), .first, (.firstFactor 17), .first, .first, .first, .first, .first, .second, .first, .first, (.secondFactor 13), (.secondFactor 23), .first, (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, (.secondFactor 31), (.firstFactor 17), .first, .first, (.firstFactor 7), .second, .first, .first, .first, .second, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row9661_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 9661 row9661_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan9661_block_008
  · decide
  · rfl

def row9661_block_009 : List Nat := []
theorem row9661_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 9661 row9661_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row9661_block_010 : List Nat := []
theorem row9661_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 9661 row9661_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row9661_block_011 : List Nat := []
theorem row9661_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 9661 row9661_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row9661_block_012 : List Nat := []
theorem row9661_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 9661 row9661_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row9661_block_013 : List Nat := []
theorem row9661_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 9661 row9661_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row9661_large_left_left : List Nat := row9661_small ++ row9661_block_005
theorem row9661_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 9661 row9661_large_left_left :=
  Budget.RowUpperBound.branch row9661_small_covers row9661_block_005_covers

def row9661_large_left_right_right : List Nat := row9661_block_007 ++ row9661_block_008
theorem row9661_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 9661 row9661_large_left_right_right :=
  Budget.RowUpperBound.branch row9661_block_007_covers row9661_block_008_covers

def row9661_large_left_right : List Nat := row9661_block_006 ++ row9661_large_left_right_right
theorem row9661_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 9661 row9661_large_left_right :=
  Budget.RowUpperBound.branch row9661_block_006_covers row9661_large_left_right_right_covers

def row9661_large_left : List Nat := row9661_large_left_left ++ row9661_large_left_right
theorem row9661_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 9661 row9661_large_left :=
  Budget.RowUpperBound.branch row9661_large_left_left_covers row9661_large_left_right_covers

def row9661_large_right_left : List Nat := row9661_block_009 ++ row9661_block_010
theorem row9661_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 9661 row9661_large_right_left :=
  Budget.RowUpperBound.branch row9661_block_009_covers row9661_block_010_covers

def row9661_large_right_right_right : List Nat := row9661_block_012 ++ row9661_block_013
theorem row9661_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 9661 row9661_large_right_right_right :=
  Budget.RowUpperBound.branch row9661_block_012_covers row9661_block_013_covers

def row9661_large_right_right : List Nat := row9661_block_011 ++ row9661_large_right_right_right
theorem row9661_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 9661 row9661_large_right_right :=
  Budget.RowUpperBound.branch row9661_block_011_covers row9661_large_right_right_right_covers

def row9661_large_right : List Nat := row9661_large_right_left ++ row9661_large_right_right
theorem row9661_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 9661 row9661_large_right :=
  Budget.RowUpperBound.branch row9661_large_right_left_covers row9661_large_right_right_covers

def row9661_large : List Nat := row9661_large_left ++ row9661_large_right
theorem row9661_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 9661 row9661_large :=
  Budget.RowUpperBound.branch row9661_large_left_covers row9661_large_right_covers

def row9661 : List Nat := row9661_large
theorem row9661_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 9661 row9661 := row9661_large_covers

def row9677_small : List Nat := []
theorem row9677_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 9677 row9677_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row9677_block_005 : List Nat := []
def plan9677_block_005 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, (.secondFactor 7), .first, (.secondFactor 29), .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 7), .first, .first, .first, (.firstFactor 23), (.secondFactor 19), .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 7), (.secondFactor 7), .first, .first, .first, (.secondFactor 13), .first, .first, (.firstFactor 31), (.firstFactor 17), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 17), .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 11), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 17), (.firstFactor 13), (.firstFactor 19), .first, (.firstFactor 7), .second, (.secondFactor 19), .first, (.firstFactor 11), .first, (.firstFactor 7), (.secondFactor 17), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), (.secondFactor 11), .first, .first]
set_option maxRecDepth 2048 in
theorem row9677_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 9677 row9677_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan9677_block_005
  · decide
  · rfl

def row9677_block_006 : List Nat := [11777, 11813, 11897, 11909]
def plan9677_block_006 : List Budget.Choice := [.first, .first, (.secondFactor 23), .first, (.secondFactor 13), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 23), (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 11), (.secondFactor 11), (.secondFactor 7), .first, (.secondFactor 31), .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 37), (.firstFactor 7), .first, .first, .first, (.firstFactor 11), (.secondFactor 7), .first, .first, (.secondFactor 7), .first, (.firstFactor 11), .first, .first, .second, .first, .first, .first, .first, .second, (.firstFactor 29), .first, .first, .first, (.secondFactor 11), .first, (.secondFactor 7), .first, (.firstFactor 31), (.secondFactor 13), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 13), .first, (.firstFactor 13), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 31), .first, (.firstFactor 7), .first, (.firstFactor 23), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.firstFactor 11), .second, .first, .first, .first, .first, .first, (.secondFactor 17), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 11), (.secondFactor 11), (.firstFactor 29), (.firstFactor 7), .first, .first, (.secondFactor 13), .first, .first, .first, .second, .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 7), .first, (.firstFactor 37), .first, .first, (.secondFactor 11), .first, (.firstFactor 17), (.firstFactor 7), (.firstFactor 11), .first, (.secondFactor 13), .first, .second, .first, (.firstFactor 7), .first, .first, .first, .possible, .first, .second, (.firstFactor 11), (.firstFactor 7), (.secondFactor 23), .possible, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 23), .first, .possible, (.firstFactor 31), .possible, .first, (.firstFactor 7), (.firstFactor 19), .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 29), (.firstFactor 11), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.secondFactor 13), .first, .first, .second, (.firstFactor 29), .first, .first, (.firstFactor 11), (.firstFactor 23), (.secondFactor 13), .first, (.firstFactor 13), .first, (.secondFactor 19), .first, .first, (.secondFactor 13), (.firstFactor 13), .first, (.firstFactor 11), .first, (.firstFactor 7), .second, .first, (.firstFactor 23), .first, .first, (.secondFactor 11), (.secondFactor 29), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, .first, .second]
set_option maxRecDepth 2048 in
theorem row9677_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 9677 row9677_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan9677_block_006
  · decide
  · rfl

def row9677_block_007 : List Nat := [13121]
def plan9677_block_007 : List Budget.Choice := [.first, .first, (.firstFactor 17), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 31), (.secondFactor 23), (.firstFactor 19), .first, .first, .second, (.firstFactor 17), .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 11), (.secondFactor 11), .first, (.secondFactor 7), (.firstFactor 37), .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 17), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .second, .first, (.firstFactor 11), (.secondFactor 11), .first, (.secondFactor 7), (.firstFactor 13), (.secondFactor 17), (.firstFactor 19), .first, .first, .first, (.secondFactor 11), .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 11), (.firstFactor 17), .first, .first, .first, (.firstFactor 7), (.firstFactor 11), .possible, .first, .first, (.firstFactor 17), .first, (.secondFactor 13), .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 11), .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 13), (.secondFactor 19), .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 31), .first, (.secondFactor 19), (.firstFactor 17), (.secondFactor 17), (.secondFactor 7), .first, (.firstFactor 13), (.firstFactor 29), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 31), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 11), .first, (.firstFactor 13), .first, .first, .second, (.firstFactor 19), .first, (.firstFactor 17), .first, .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 11), (.firstFactor 7), .first, (.firstFactor 13), .first, .first, (.firstFactor 7), .first, (.secondFactor 11), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 7), .first, .first, .first, .second, .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 13), .first, .first, .first, (.firstFactor 37), (.secondFactor 19), .first, .first, .first, (.secondFactor 11), (.firstFactor 7), .first, (.firstFactor 19), .first, .first, (.firstFactor 29), .first, .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 13), .first, .first, (.firstFactor 7), .first, .first]
set_option maxRecDepth 2048 in
theorem row9677_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 9677 row9677_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan9677_block_007
  · decide
  · rfl

def row9677_block_008 : List Nat := [14639, 15017, 15137, 16097]
def plan9677_block_008 : List Budget.Choice := [.first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.secondFactor 31), (.firstFactor 29), .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 19), .first, .first, (.firstFactor 11), .first, (.secondFactor 11), .possible, .first, (.firstFactor 7), (.firstFactor 19), .first, (.firstFactor 7), .first, (.firstFactor 29), (.secondFactor 13), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 13), .first, .first, (.secondFactor 11), (.secondFactor 23), .first, (.firstFactor 7), .first, (.firstFactor 31), .first, (.firstFactor 11), (.firstFactor 19), .first, .first, (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 11), (.firstFactor 13), .first, .first, .possible, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 37), .possible, .first, (.secondFactor 7), (.firstFactor 7), .second, .first, .first, .first, .first, (.secondFactor 11), (.secondFactor 7), .first, .first, .second, (.secondFactor 13), .first, .first, (.firstFactor 7), .first, (.secondFactor 17), .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 13), .first, .first, .second, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, (.secondFactor 7), .first, (.secondFactor 29), (.firstFactor 19), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 11), .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 11), .first, (.firstFactor 23), .first, (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 29), .first, .first, .first, (.firstFactor 7), (.firstFactor 37), (.firstFactor 17), (.firstFactor 19), .first, .first, .first, .first, (.firstFactor 11), (.secondFactor 11), .first, .first, .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 7), (.secondFactor 13), .first, .first, (.firstFactor 7), (.firstFactor 17), .first, .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 7), .first, .first, .possible, .second, .first, (.firstFactor 7), (.firstFactor 13), .first, .first, (.secondFactor 19), .first, .first, (.firstFactor 13), (.secondFactor 23), .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 19), (.secondFactor 17), .first, .first, (.secondFactor 11), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row9677_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 9677 row9677_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan9677_block_008
  · decide
  · rfl

def row9677_block_009 : List Nat := []
theorem row9677_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 9677 row9677_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row9677_block_010 : List Nat := []
theorem row9677_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 9677 row9677_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row9677_block_011 : List Nat := []
theorem row9677_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 9677 row9677_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row9677_block_012 : List Nat := []
theorem row9677_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 9677 row9677_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row9677_block_013 : List Nat := []
theorem row9677_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 9677 row9677_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row9677_large_left_left : List Nat := row9677_small ++ row9677_block_005
theorem row9677_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 9677 row9677_large_left_left :=
  Budget.RowUpperBound.branch row9677_small_covers row9677_block_005_covers

def row9677_large_left_right_right : List Nat := row9677_block_007 ++ row9677_block_008
theorem row9677_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 9677 row9677_large_left_right_right :=
  Budget.RowUpperBound.branch row9677_block_007_covers row9677_block_008_covers

def row9677_large_left_right : List Nat := row9677_block_006 ++ row9677_large_left_right_right
theorem row9677_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 9677 row9677_large_left_right :=
  Budget.RowUpperBound.branch row9677_block_006_covers row9677_large_left_right_right_covers

def row9677_large_left : List Nat := row9677_large_left_left ++ row9677_large_left_right
theorem row9677_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 9677 row9677_large_left :=
  Budget.RowUpperBound.branch row9677_large_left_left_covers row9677_large_left_right_covers

def row9677_large_right_left : List Nat := row9677_block_009 ++ row9677_block_010
theorem row9677_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 9677 row9677_large_right_left :=
  Budget.RowUpperBound.branch row9677_block_009_covers row9677_block_010_covers

def row9677_large_right_right_right : List Nat := row9677_block_012 ++ row9677_block_013
theorem row9677_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 9677 row9677_large_right_right_right :=
  Budget.RowUpperBound.branch row9677_block_012_covers row9677_block_013_covers

def row9677_large_right_right : List Nat := row9677_block_011 ++ row9677_large_right_right_right
theorem row9677_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 9677 row9677_large_right_right :=
  Budget.RowUpperBound.branch row9677_block_011_covers row9677_large_right_right_right_covers

def row9677_large_right : List Nat := row9677_large_right_left ++ row9677_large_right_right
theorem row9677_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 9677 row9677_large_right :=
  Budget.RowUpperBound.branch row9677_large_right_left_covers row9677_large_right_right_covers

def row9677_large : List Nat := row9677_large_left ++ row9677_large_right
theorem row9677_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 9677 row9677_large :=
  Budget.RowUpperBound.branch row9677_large_left_covers row9677_large_right_covers

def row9677 : List Nat := row9677_large
theorem row9677_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 9677 row9677 := row9677_large_covers

end ProjectEulerSolutions.PrimePairLarge
