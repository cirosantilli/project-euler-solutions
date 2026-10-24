import ProjectEulerSolutions.Lib.PrimePairLarge.Bounds

namespace ProjectEulerSolutions.PrimePairLarge
open PrimePairCertificate

def row10651_small : List Nat := []
theorem row10651_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 10651 row10651_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row10651_block_005 : List Nat := []
theorem row10651_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 10651 row10651_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row10651_block_006 : List Nat := [10939, 11197, 11689]
def plan10651_block_006 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, (.firstFactor 13), (.secondFactor 7), .first, (.secondFactor 19), .first, .first, (.firstFactor 7), .first, (.secondFactor 29), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 13), .first, .first, (.firstFactor 13), (.secondFactor 29), .first, .first, .possible, .first, (.secondFactor 7), .first, .first, (.firstFactor 19), .first, .first, .first, (.firstFactor 7), .first, .second, .first, (.firstFactor 23), (.secondFactor 7), .first, .first, (.firstFactor 11), .first, (.firstFactor 37), (.firstFactor 7), (.firstFactor 17), .first, .first, .first, (.firstFactor 7), .first, .possible, .first, .second, .first, (.firstFactor 17), (.firstFactor 7), .first, .first, .first, (.secondFactor 31), (.firstFactor 7), (.firstFactor 11), .first, .first, (.firstFactor 19), .first, (.firstFactor 17), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 13), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 7), .first, .second, .first, .first, .first, (.firstFactor 7), .first, .possible, .first, .first, .first, (.firstFactor 7), .first, .second, .first, (.secondFactor 37), .first, .first, .first, .first, .first, .second, (.firstFactor 13), .first, .second, (.firstFactor 11), (.firstFactor 17), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 17), .first, (.secondFactor 13), .first, (.firstFactor 7), .first, .first, (.firstFactor 37), .first, (.firstFactor 11), .first, (.firstFactor 31), (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .second, .first, .first, (.firstFactor 19), .first, .first, .first, .first, .first, .first, .second, (.firstFactor 11), .first, .first, (.secondFactor 7), .first, (.firstFactor 13), .first, .first, (.firstFactor 7), .first, (.secondFactor 23), .first]
set_option maxRecDepth 2048 in
theorem row10651_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 10651 row10651_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan10651_block_006
  · decide
  · rfl

def row10651_block_007 : List Nat := [12613, 13249, 13963]
def plan10651_block_007 : List Budget.Choice := [.first, (.firstFactor 7), .first, (.firstFactor 13), .second, .first, .first, .second, .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, (.secondFactor 23), (.secondFactor 7), .first, (.firstFactor 17), (.firstFactor 19), (.firstFactor 23), (.firstFactor 7), .first, .possible, .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 11), .second, .first, (.secondFactor 7), (.secondFactor 13), .first, (.secondFactor 17), (.firstFactor 11), (.firstFactor 17), .first, .first, .first, .first, (.secondFactor 23), (.firstFactor 11), (.firstFactor 13), (.firstFactor 7), (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 13), .first, .first, .first, .first, (.secondFactor 19), (.firstFactor 31), (.firstFactor 7), .first, .first, (.firstFactor 23), .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 11), (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, .possible, .first, (.firstFactor 19), (.firstFactor 11), .second, (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 37), .first, (.firstFactor 19), .first, (.firstFactor 7), .second, (.firstFactor 23), .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 23), .first, (.firstFactor 29), .first, (.firstFactor 7), .first, .first, (.firstFactor 17), .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, (.firstFactor 29), .first, (.firstFactor 19), (.secondFactor 7), .first, .first, .first, .first, .first, (.secondFactor 23), .first, .first, .first, (.firstFactor 23), .first, .first, .first, .first, (.secondFactor 17), .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 13), .possible, .first, .first, (.secondFactor 37), .first, (.firstFactor 13), (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 17), (.secondFactor 13), (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 13), .first, .first, .second, (.firstFactor 7), (.firstFactor 19), .first, .first, (.firstFactor 7), .first, (.secondFactor 23), (.firstFactor 11), .first, .first, (.secondFactor 17), (.firstFactor 13), (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 31)]
set_option maxRecDepth 2048 in
theorem row10651_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 10651 row10651_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan10651_block_007
  · decide
  · rfl

def row10651_block_008 : List Nat := []
def plan10651_block_008 : List Budget.Choice := [(.firstFactor 11), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 13), .first, (.secondFactor 19), .first, (.secondFactor 17), .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 13), .first, .first, (.secondFactor 29), (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 13), .first, (.secondFactor 7), .first, (.firstFactor 23), .first, (.firstFactor 17), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 37), .first, .first, (.firstFactor 17), (.firstFactor 29), .first, (.firstFactor 13), .first, .first, .first, .second, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 19), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 31), (.secondFactor 7), .first, .first, .first, (.secondFactor 7), (.firstFactor 13), .first, .first, (.firstFactor 11), (.firstFactor 29), .first, (.firstFactor 7), .first, (.secondFactor 17), (.secondFactor 13), (.firstFactor 19), .first, (.firstFactor 7), (.firstFactor 23), .first, (.secondFactor 19), (.firstFactor 7), .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row10651_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 10651 row10651_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan10651_block_008
  · decide
  · rfl

def row10651_block_009 : List Nat := []
theorem row10651_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 10651 row10651_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row10651_block_010 : List Nat := []
theorem row10651_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 10651 row10651_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row10651_block_011 : List Nat := []
theorem row10651_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 10651 row10651_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row10651_block_012 : List Nat := []
theorem row10651_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 10651 row10651_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row10651_block_013 : List Nat := []
theorem row10651_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 10651 row10651_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row10651_large_left_left : List Nat := row10651_small ++ row10651_block_005
theorem row10651_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 10651 row10651_large_left_left :=
  Budget.RowUpperBound.branch row10651_small_covers row10651_block_005_covers

def row10651_large_left_right_right : List Nat := row10651_block_007 ++ row10651_block_008
theorem row10651_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 10651 row10651_large_left_right_right :=
  Budget.RowUpperBound.branch row10651_block_007_covers row10651_block_008_covers

def row10651_large_left_right : List Nat := row10651_block_006 ++ row10651_large_left_right_right
theorem row10651_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 10651 row10651_large_left_right :=
  Budget.RowUpperBound.branch row10651_block_006_covers row10651_large_left_right_right_covers

def row10651_large_left : List Nat := row10651_large_left_left ++ row10651_large_left_right
theorem row10651_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 10651 row10651_large_left :=
  Budget.RowUpperBound.branch row10651_large_left_left_covers row10651_large_left_right_covers

def row10651_large_right_left : List Nat := row10651_block_009 ++ row10651_block_010
theorem row10651_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 10651 row10651_large_right_left :=
  Budget.RowUpperBound.branch row10651_block_009_covers row10651_block_010_covers

def row10651_large_right_right_right : List Nat := row10651_block_012 ++ row10651_block_013
theorem row10651_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 10651 row10651_large_right_right_right :=
  Budget.RowUpperBound.branch row10651_block_012_covers row10651_block_013_covers

def row10651_large_right_right : List Nat := row10651_block_011 ++ row10651_large_right_right_right
theorem row10651_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 10651 row10651_large_right_right :=
  Budget.RowUpperBound.branch row10651_block_011_covers row10651_large_right_right_right_covers

def row10651_large_right : List Nat := row10651_large_right_left ++ row10651_large_right_right
theorem row10651_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 10651 row10651_large_right :=
  Budget.RowUpperBound.branch row10651_large_right_left_covers row10651_large_right_right_covers

def row10651_large : List Nat := row10651_large_left ++ row10651_large_right
theorem row10651_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 10651 row10651_large :=
  Budget.RowUpperBound.branch row10651_large_left_covers row10651_large_right_covers

def row10651 : List Nat := row10651_large
theorem row10651_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 10651 row10651 := row10651_large_covers

def row10657_small : List Nat := []
theorem row10657_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 10657 row10657_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row10657_block_005 : List Nat := []
theorem row10657_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 10657 row10657_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row10657_block_006 : List Nat := [11173, 11677, 11971]
def plan10657_block_006 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, (.secondFactor 29), .first, (.secondFactor 7), .first, .first, (.firstFactor 13), (.firstFactor 7), (.secondFactor 7), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 11), .first, (.secondFactor 37), (.secondFactor 19), .first, .first, .first, .second, (.firstFactor 13), .first, .first, (.firstFactor 7), (.secondFactor 17), (.firstFactor 17), .first, (.secondFactor 7), .first, (.firstFactor 23), .first, .first, (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 31), (.firstFactor 19), .first, .first, (.firstFactor 17), .first, (.firstFactor 11), .first, (.firstFactor 31), .first, .first, .first, .possible, .first, (.firstFactor 19), .first, .first, .first, (.firstFactor 11), (.firstFactor 13), .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 11), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .second, .first, .first, .first, (.firstFactor 13), (.secondFactor 37), .first, .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 13), .first, .first, (.secondFactor 17), .first, .first, .first, .possible, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 17), (.firstFactor 7), (.secondFactor 13), .first, (.firstFactor 11), .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 17), .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 23), .first, .first, .first, (.firstFactor 7), .second, (.firstFactor 13), .first, .possible, .first, .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 11), .first, .first, (.firstFactor 37), (.firstFactor 29), .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 37), .first, .first, (.secondFactor 13), .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 13), .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 7), .first]
set_option maxRecDepth 2048 in
theorem row10657_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 10657 row10657_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan10657_block_006
  · decide
  · rfl

def row10657_block_007 : List Nat := [12589, 12919, 12979, 13681, 14437]
def plan10657_block_007 : List Budget.Choice := [.second, (.secondFactor 17), .first, (.firstFactor 19), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 37), .first, .first, (.firstFactor 17), (.secondFactor 19), .first, .first, (.secondFactor 7), (.firstFactor 13), .possible, (.secondFactor 13), .first, (.firstFactor 7), (.firstFactor 29), (.firstFactor 11), .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 11), .first, (.secondFactor 23), (.firstFactor 7), .first, (.secondFactor 13), .first, (.firstFactor 7), .first, .second, .first, .first, (.firstFactor 7), (.secondFactor 7), (.secondFactor 17), .first, (.firstFactor 23), .first, .first, (.firstFactor 7), .first, .first, .possible, .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 13), .possible, .first, .first, (.secondFactor 19), .first, (.firstFactor 31), (.firstFactor 7), .first, .first, .first, .second, .second, (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 7), .first, .first, .second, (.firstFactor 37), .first, .first, .first, .first, .first, (.firstFactor 19), .first, .first, (.secondFactor 7), (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 31), .first, .first, (.firstFactor 7), (.secondFactor 7), .first, (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 19), .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 31), .first, (.firstFactor 19), (.firstFactor 13), .first, .first, (.firstFactor 11), .second, .first, (.secondFactor 7), .first, .possible, (.secondFactor 19), .first, (.firstFactor 11), .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 19), (.secondFactor 7), .first, .first, (.firstFactor 7), .first, (.firstFactor 29), (.secondFactor 19), .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 7), .first, .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 17), .first, .second, .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .second, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 19), .second, (.firstFactor 23), .first, (.firstFactor 7), .first, .second, .possible, .first, .first, (.firstFactor 7)]
set_option maxRecDepth 2048 in
theorem row10657_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 10657 row10657_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan10657_block_007
  · decide
  · rfl

def row10657_block_008 : List Nat := []
def plan10657_block_008 : List Budget.Choice := [(.firstFactor 17), .first, (.firstFactor 7), .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 23), .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, (.firstFactor 19), (.firstFactor 37), .first, .first, .first, .first, (.firstFactor 13), .first, (.secondFactor 17), .first, (.firstFactor 7), .first, (.firstFactor 23), (.secondFactor 19), .first, .first, .second, .first, (.firstFactor 31), .first, (.firstFactor 17), .first, .first, (.firstFactor 7), (.secondFactor 7), .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 17), (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 19), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 19), (.firstFactor 17), .first, (.firstFactor 7), .first, .first, (.firstFactor 31), (.firstFactor 7), .first, .first, (.secondFactor 37), (.firstFactor 11), .first, (.secondFactor 17), .first, (.secondFactor 7), (.firstFactor 13), .first, .first, (.secondFactor 13), (.secondFactor 7), .first, (.secondFactor 29), (.firstFactor 23), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row10657_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 10657 row10657_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan10657_block_008
  · decide
  · rfl

def row10657_block_009 : List Nat := []
theorem row10657_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 10657 row10657_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row10657_block_010 : List Nat := []
theorem row10657_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 10657 row10657_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row10657_block_011 : List Nat := []
theorem row10657_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 10657 row10657_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row10657_block_012 : List Nat := []
theorem row10657_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 10657 row10657_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row10657_block_013 : List Nat := []
theorem row10657_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 10657 row10657_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row10657_large_left_left : List Nat := row10657_small ++ row10657_block_005
theorem row10657_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 10657 row10657_large_left_left :=
  Budget.RowUpperBound.branch row10657_small_covers row10657_block_005_covers

def row10657_large_left_right_right : List Nat := row10657_block_007 ++ row10657_block_008
theorem row10657_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 10657 row10657_large_left_right_right :=
  Budget.RowUpperBound.branch row10657_block_007_covers row10657_block_008_covers

def row10657_large_left_right : List Nat := row10657_block_006 ++ row10657_large_left_right_right
theorem row10657_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 10657 row10657_large_left_right :=
  Budget.RowUpperBound.branch row10657_block_006_covers row10657_large_left_right_right_covers

def row10657_large_left : List Nat := row10657_large_left_left ++ row10657_large_left_right
theorem row10657_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 10657 row10657_large_left :=
  Budget.RowUpperBound.branch row10657_large_left_left_covers row10657_large_left_right_covers

def row10657_large_right_left : List Nat := row10657_block_009 ++ row10657_block_010
theorem row10657_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 10657 row10657_large_right_left :=
  Budget.RowUpperBound.branch row10657_block_009_covers row10657_block_010_covers

def row10657_large_right_right_right : List Nat := row10657_block_012 ++ row10657_block_013
theorem row10657_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 10657 row10657_large_right_right_right :=
  Budget.RowUpperBound.branch row10657_block_012_covers row10657_block_013_covers

def row10657_large_right_right : List Nat := row10657_block_011 ++ row10657_large_right_right_right
theorem row10657_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 10657 row10657_large_right_right :=
  Budget.RowUpperBound.branch row10657_block_011_covers row10657_large_right_right_right_covers

def row10657_large_right : List Nat := row10657_large_right_left ++ row10657_large_right_right
theorem row10657_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 10657 row10657_large_right :=
  Budget.RowUpperBound.branch row10657_large_right_left_covers row10657_large_right_right_covers

def row10657_large : List Nat := row10657_large_left ++ row10657_large_right
theorem row10657_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 10657 row10657_large :=
  Budget.RowUpperBound.branch row10657_large_left_covers row10657_large_right_covers

def row10657 : List Nat := row10657_large
theorem row10657_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 10657 row10657 := row10657_large_covers

def row10663_small : List Nat := []
theorem row10663_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 10663 row10663_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row10663_block_005 : List Nat := []
theorem row10663_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 10663 row10663_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row10663_block_006 : List Nat := [10909, 10939, 11161]
def plan10663_block_006 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, (.firstFactor 13), .first, .first, (.secondFactor 7), (.firstFactor 19), (.firstFactor 11), .first, .first, (.secondFactor 7), (.firstFactor 17), .first, .first, .first, (.firstFactor 29), (.firstFactor 19), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .possible, .first, .possible, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 11), .first, .first, (.secondFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 23), (.secondFactor 7), .first, .first, .possible, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 37), (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 13), .first, .first, (.secondFactor 19), .first, (.firstFactor 29), .first, (.firstFactor 17), .first, .first, .first, .first, (.secondFactor 37), (.secondFactor 19), .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 7), (.secondFactor 17), .first, .first, (.firstFactor 29), .first, (.secondFactor 7), .first, (.firstFactor 11), (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 17), .first, (.firstFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 7), .first, (.firstFactor 13), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 13), (.firstFactor 19), .first, (.secondFactor 7), .first, .first, .first, .second, .first, .first, .first, .second, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 17), .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, (.secondFactor 13), (.firstFactor 17), .first, .first, (.secondFactor 17), .first, .first, .first, .first, (.secondFactor 31), .first, (.firstFactor 7), .first, .first, (.secondFactor 17), .first, .first, (.secondFactor 19), .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 17), .first, .first, (.secondFactor 23), .first, (.firstFactor 7), .first, (.firstFactor 11), (.secondFactor 7), .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row10663_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 10663 row10663_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan10663_block_006
  · decide
  · rfl

def row10663_block_007 : List Nat := [13099, 13291, 13873, 14461]
def plan10663_block_007 : List Budget.Choice := [.first, (.firstFactor 19), .first, .second, (.firstFactor 7), .first, .first, (.firstFactor 31), .first, .first, .first, (.firstFactor 11), (.secondFactor 7), .first, .first, (.firstFactor 7), (.firstFactor 19), .first, .first, (.firstFactor 11), (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.secondFactor 13), (.firstFactor 13), .first, .first, .first, .first, .first, .first, (.secondFactor 13), .first, .first, (.secondFactor 17), .first, .first, (.secondFactor 23), .second, .first, .first, .second, .first, .first, (.secondFactor 17), .first, (.firstFactor 11), (.secondFactor 7), (.firstFactor 19), .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 11), (.secondFactor 7), .first, .first, (.firstFactor 7), .first, (.secondFactor 13), (.secondFactor 23), .first, .first, .first, (.secondFactor 7), (.firstFactor 29), .possible, .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 37), .first, (.firstFactor 29), .possible, (.firstFactor 7), (.secondFactor 23), .first, (.firstFactor 23), .first, .first, (.firstFactor 7), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 13), .first, (.firstFactor 29), .first, .first, .first, .first, (.secondFactor 13), .first, .first, .first, .first, (.secondFactor 17), .first, (.firstFactor 11), .first, (.firstFactor 7), .second, .first, .first, (.firstFactor 17), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 37), .first, .first, (.secondFactor 13), .first, (.secondFactor 19), (.firstFactor 13), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 29), .first, (.firstFactor 13), .first, (.firstFactor 11), .first, .first, .possible, .first, (.firstFactor 23), .first, .first, (.secondFactor 7), .first, .first, .second, .first, (.firstFactor 17), (.firstFactor 11), .first, .first, (.secondFactor 23), .first, (.firstFactor 7), (.firstFactor 11), .first, .first, .first, (.secondFactor 7), .first, .second, .first, .second, (.firstFactor 19), (.secondFactor 17), .first, .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, (.secondFactor 17), (.secondFactor 7), (.firstFactor 11), .first, .first, (.secondFactor 7), .first, (.firstFactor 17), (.firstFactor 7), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .second, .first, (.firstFactor 7), .second, .first, (.secondFactor 7), .possible]
set_option maxRecDepth 2048 in
theorem row10663_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 10663 row10663_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan10663_block_007
  · decide
  · rfl

def row10663_block_008 : List Nat := [14923, 15031, 15199]
def plan10663_block_008 : List Budget.Choice := [.first, .first, .second, .first, (.firstFactor 31), .first, .first, .first, (.secondFactor 23), (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 19), .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .second, .first, (.firstFactor 11), (.firstFactor 19), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 11), .first, .first, .possible, .first, .first, (.secondFactor 29), .first, .first, .first, (.firstFactor 23), .first, .first, .possible, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 31), .first, .first, (.firstFactor 23), .first, .first, .second, .first, .first, .first, (.firstFactor 7), (.secondFactor 13), .possible, (.firstFactor 11), .first, .first, (.secondFactor 23), (.firstFactor 17), .first, .first, (.firstFactor 7), (.firstFactor 31), .first, (.firstFactor 13), .first, .first, (.firstFactor 7), (.secondFactor 19), .first, (.secondFactor 7), (.firstFactor 11), .first, (.firstFactor 17), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row10663_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 10663 row10663_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan10663_block_008
  · decide
  · rfl

def row10663_block_009 : List Nat := []
theorem row10663_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 10663 row10663_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row10663_block_010 : List Nat := []
theorem row10663_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 10663 row10663_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row10663_block_011 : List Nat := []
theorem row10663_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 10663 row10663_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row10663_block_012 : List Nat := []
theorem row10663_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 10663 row10663_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row10663_block_013 : List Nat := []
theorem row10663_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 10663 row10663_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row10663_large_left_left : List Nat := row10663_small ++ row10663_block_005
theorem row10663_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 10663 row10663_large_left_left :=
  Budget.RowUpperBound.branch row10663_small_covers row10663_block_005_covers

def row10663_large_left_right_right : List Nat := row10663_block_007 ++ row10663_block_008
theorem row10663_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 10663 row10663_large_left_right_right :=
  Budget.RowUpperBound.branch row10663_block_007_covers row10663_block_008_covers

def row10663_large_left_right : List Nat := row10663_block_006 ++ row10663_large_left_right_right
theorem row10663_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 10663 row10663_large_left_right :=
  Budget.RowUpperBound.branch row10663_block_006_covers row10663_large_left_right_right_covers

def row10663_large_left : List Nat := row10663_large_left_left ++ row10663_large_left_right
theorem row10663_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 10663 row10663_large_left :=
  Budget.RowUpperBound.branch row10663_large_left_left_covers row10663_large_left_right_covers

def row10663_large_right_left : List Nat := row10663_block_009 ++ row10663_block_010
theorem row10663_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 10663 row10663_large_right_left :=
  Budget.RowUpperBound.branch row10663_block_009_covers row10663_block_010_covers

def row10663_large_right_right_right : List Nat := row10663_block_012 ++ row10663_block_013
theorem row10663_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 10663 row10663_large_right_right_right :=
  Budget.RowUpperBound.branch row10663_block_012_covers row10663_block_013_covers

def row10663_large_right_right : List Nat := row10663_block_011 ++ row10663_large_right_right_right
theorem row10663_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 10663 row10663_large_right_right :=
  Budget.RowUpperBound.branch row10663_block_011_covers row10663_large_right_right_right_covers

def row10663_large_right : List Nat := row10663_large_right_left ++ row10663_large_right_right
theorem row10663_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 10663 row10663_large_right :=
  Budget.RowUpperBound.branch row10663_large_right_left_covers row10663_large_right_right_covers

def row10663_large : List Nat := row10663_large_left ++ row10663_large_right
theorem row10663_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 10663 row10663_large :=
  Budget.RowUpperBound.branch row10663_large_left_covers row10663_large_right_covers

def row10663 : List Nat := row10663_large
theorem row10663_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 10663 row10663 := row10663_large_covers

def row10667_small : List Nat := []
theorem row10667_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 10667 row10667_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row10667_block_005 : List Nat := []
theorem row10667_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 10667 row10667_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row10667_block_006 : List Nat := [10691, 11351, 12161]
def plan10667_block_006 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, .possible, .second, .first, .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 13), (.firstFactor 23), .first, .first, (.firstFactor 7), (.secondFactor 13), .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 29), (.secondFactor 7), .first, .first, .second, (.secondFactor 37), .first, .first, .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 7), .first, .second, .first, .first, .first, .first, .first, (.secondFactor 23), .first, (.firstFactor 7), .first, (.secondFactor 17), .first, (.firstFactor 13), .first, .first, (.firstFactor 7), (.firstFactor 23), (.secondFactor 13), .first, .first, .first, .first, (.firstFactor 13), .first, .possible, .first, (.firstFactor 37), .first, (.firstFactor 11), (.firstFactor 13), (.firstFactor 23), (.secondFactor 29), .first, .first, (.secondFactor 23), .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 31), .first, (.firstFactor 23), .first, (.secondFactor 31), .first, .first, (.firstFactor 7), .first, (.secondFactor 17), (.firstFactor 13), (.firstFactor 11), .first, (.firstFactor 7), .first, (.firstFactor 17), .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 11), (.firstFactor 17), (.firstFactor 7), (.firstFactor 19), .first, .first, .first, .first, .first, .first, (.firstFactor 13), .first, .second, (.firstFactor 17), .first, .first, (.firstFactor 19), (.firstFactor 7), .first, .first, .first, .first, .second, .first, (.secondFactor 13), (.firstFactor 11), .first, .first, .first, (.firstFactor 19), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 7), (.firstFactor 17), .first, (.secondFactor 7), (.firstFactor 11), (.firstFactor 7), .first, .first, .possible, .first, (.secondFactor 7), .first, .first, (.firstFactor 7), (.firstFactor 23), .first, (.firstFactor 11), .first, (.firstFactor 31), (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 23), .first, .first, .first, .first, (.firstFactor 13)]
set_option maxRecDepth 2048 in
theorem row10667_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 10667 row10667_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan10667_block_006
  · decide
  · rfl

def row10667_block_007 : List Nat := [13397]
def plan10667_block_007 : List Budget.Choice := [.first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 13), (.firstFactor 19), .first, .first, .first, (.secondFactor 13), (.firstFactor 29), .first, .first, .first, (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 19), .first, .first, .first, (.secondFactor 17), (.firstFactor 7), (.firstFactor 23), (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 23), .first, .first, (.firstFactor 17), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), (.secondFactor 13), .first, (.firstFactor 17), (.firstFactor 7), (.firstFactor 19), (.firstFactor 13), .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 31), .first, .first, (.firstFactor 13), (.firstFactor 11), (.secondFactor 17), .first, .first, .first, (.secondFactor 23), (.firstFactor 7), (.secondFactor 7), (.firstFactor 17), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 17), (.firstFactor 11), .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 17), (.secondFactor 29), .first, (.firstFactor 37), .first, .possible, .first, .first, .first, .first, .first, .first, (.secondFactor 7), (.secondFactor 13), (.secondFactor 37), .first, (.firstFactor 7), (.secondFactor 7), .first, (.firstFactor 19), .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 23), .first, .first, .first, .first, .first, .first, .first, (.secondFactor 37), .first, (.firstFactor 7), (.secondFactor 7), .first, .second, .first, .first, (.firstFactor 19), (.firstFactor 23), .first, (.secondFactor 17), (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 17), (.secondFactor 29), .first, (.secondFactor 7), .first, .first, (.firstFactor 11), .first, (.firstFactor 7), (.secondFactor 37), .first, (.firstFactor 29), .first, .first, (.firstFactor 11), (.secondFactor 31), .first, (.secondFactor 13), .first, .first, (.firstFactor 7), (.firstFactor 13), .second, .first, .second, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 7), (.firstFactor 17), .first, .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 7), .first, .first, (.firstFactor 7), .second, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.firstFactor 23), .first, .first]
set_option maxRecDepth 2048 in
theorem row10667_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 10667 row10667_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan10667_block_007
  · decide
  · rfl

def row10667_block_008 : List Nat := [14669, 14771]
def plan10667_block_008 : List Budget.Choice := [.first, .first, .first, (.firstFactor 13), .first, (.firstFactor 7), .first, (.firstFactor 19), .first, .first, (.firstFactor 11), .first, (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 11), .first, (.secondFactor 7), .second, .first, (.firstFactor 17), .possible, .first, (.firstFactor 37), .first, (.secondFactor 7), (.firstFactor 23), .first, .first, (.secondFactor 31), (.firstFactor 7), (.firstFactor 13), (.firstFactor 11), .first, .possible, .first, (.secondFactor 17), .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, (.firstFactor 31), .first, .first, .first, (.firstFactor 11), (.secondFactor 23), .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 31), .first, .first, (.secondFactor 19), (.firstFactor 7), .first, (.secondFactor 13), .first, .first, .first, (.firstFactor 23), .first, (.firstFactor 29), .first, (.secondFactor 23), .first, .first, .first, .first, .first, (.firstFactor 19), .first, .first, (.secondFactor 7), (.firstFactor 17), .first, .first, (.firstFactor 11), .first, (.firstFactor 13), .first, .first, .first, .first, .first, .first, .first, .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row10667_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 10667 row10667_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan10667_block_008
  · decide
  · rfl

def row10667_block_009 : List Nat := []
theorem row10667_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 10667 row10667_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row10667_block_010 : List Nat := []
theorem row10667_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 10667 row10667_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row10667_block_011 : List Nat := []
theorem row10667_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 10667 row10667_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row10667_block_012 : List Nat := []
theorem row10667_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 10667 row10667_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row10667_block_013 : List Nat := []
theorem row10667_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 10667 row10667_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row10667_large_left_left : List Nat := row10667_small ++ row10667_block_005
theorem row10667_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 10667 row10667_large_left_left :=
  Budget.RowUpperBound.branch row10667_small_covers row10667_block_005_covers

def row10667_large_left_right_right : List Nat := row10667_block_007 ++ row10667_block_008
theorem row10667_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 10667 row10667_large_left_right_right :=
  Budget.RowUpperBound.branch row10667_block_007_covers row10667_block_008_covers

def row10667_large_left_right : List Nat := row10667_block_006 ++ row10667_large_left_right_right
theorem row10667_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 10667 row10667_large_left_right :=
  Budget.RowUpperBound.branch row10667_block_006_covers row10667_large_left_right_right_covers

def row10667_large_left : List Nat := row10667_large_left_left ++ row10667_large_left_right
theorem row10667_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 10667 row10667_large_left :=
  Budget.RowUpperBound.branch row10667_large_left_left_covers row10667_large_left_right_covers

def row10667_large_right_left : List Nat := row10667_block_009 ++ row10667_block_010
theorem row10667_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 10667 row10667_large_right_left :=
  Budget.RowUpperBound.branch row10667_block_009_covers row10667_block_010_covers

def row10667_large_right_right_right : List Nat := row10667_block_012 ++ row10667_block_013
theorem row10667_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 10667 row10667_large_right_right_right :=
  Budget.RowUpperBound.branch row10667_block_012_covers row10667_block_013_covers

def row10667_large_right_right : List Nat := row10667_block_011 ++ row10667_large_right_right_right
theorem row10667_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 10667 row10667_large_right_right :=
  Budget.RowUpperBound.branch row10667_block_011_covers row10667_large_right_right_right_covers

def row10667_large_right : List Nat := row10667_large_right_left ++ row10667_large_right_right
theorem row10667_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 10667 row10667_large_right :=
  Budget.RowUpperBound.branch row10667_large_right_left_covers row10667_large_right_right_covers

def row10667_large : List Nat := row10667_large_left ++ row10667_large_right
theorem row10667_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 10667 row10667_large :=
  Budget.RowUpperBound.branch row10667_large_left_covers row10667_large_right_covers

def row10667 : List Nat := row10667_large
theorem row10667_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 10667 row10667 := row10667_large_covers

end ProjectEulerSolutions.PrimePairLarge
