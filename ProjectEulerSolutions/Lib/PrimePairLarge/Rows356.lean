import ProjectEulerSolutions.Lib.PrimePairLarge.Bounds

namespace ProjectEulerSolutions.PrimePairLarge
open PrimePairCertificate

def row11909_small : List Nat := []
theorem row11909_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 11909 row11909_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row11909_block_005 : List Nat := []
theorem row11909_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 11909 row11909_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row11909_block_006 : List Nat := []
def plan11909_block_006 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, (.secondFactor 19), (.firstFactor 37), (.firstFactor 7), .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 7), .second, .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 17), (.firstFactor 7), .first, .first, (.secondFactor 13), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 13), (.firstFactor 13), .first, .first, (.firstFactor 11), .first, (.secondFactor 7), .first, .first, (.secondFactor 19), .first, (.firstFactor 13), .first, .first, .first, .second, .first, (.firstFactor 29), .first, (.firstFactor 23), .first, .first, (.firstFactor 7), .first, (.secondFactor 23)]
set_option maxRecDepth 2048 in
theorem row11909_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 11909 row11909_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan11909_block_006
  · decide
  · rfl

def row11909_block_007 : List Nat := [12473, 13859]
def plan11909_block_007 : List Budget.Choice := [.first, .first, (.firstFactor 11), .first, .first, .possible, .first, .first, .first, (.secondFactor 19), (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 19), (.firstFactor 7), .first, (.firstFactor 13), (.secondFactor 17), .first, .first, (.firstFactor 17), .first, .first, (.secondFactor 13), .first, .first, .first, (.firstFactor 23), .first, (.secondFactor 31), (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 17), (.firstFactor 11), .first, .second, (.firstFactor 17), .first, (.secondFactor 7), .first, (.secondFactor 19), .first, .first, .first, .first, (.firstFactor 13), .second, .first, (.secondFactor 7), .first, .first, (.secondFactor 37), (.firstFactor 29), (.secondFactor 7), .first, .first, .first, (.firstFactor 19), .second, (.firstFactor 17), .first, .first, .second, .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 11), (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 19), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 29), .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, (.firstFactor 37), .second, .first, (.secondFactor 13), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, .second, .first, (.secondFactor 7), .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 37), .first, (.secondFactor 7), .first, .first, (.secondFactor 19), (.firstFactor 11), .first, (.firstFactor 13), .second, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .possible, .first, .second, .first, .first, (.firstFactor 19), .first, .first, (.firstFactor 7), .first, (.firstFactor 37), .first, .first, .first, (.firstFactor 7), .first, .second, .first, .first, (.firstFactor 23), .second, (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row11909_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 11909 row11909_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan11909_block_007
  · decide
  · rfl

def row11909_block_008 : List Nat := []
theorem row11909_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 11909 row11909_block_008 :=
  Budget.empty_row_after _ _ _ 14466 (fun q member => (Bounds.block_008 q member).1) (by decide)

def row11909_block_009 : List Nat := []
theorem row11909_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 11909 row11909_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row11909_block_010 : List Nat := []
theorem row11909_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 11909 row11909_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row11909_block_011 : List Nat := []
theorem row11909_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 11909 row11909_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row11909_block_012 : List Nat := []
theorem row11909_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 11909 row11909_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row11909_block_013 : List Nat := []
theorem row11909_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 11909 row11909_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row11909_large_left_left : List Nat := row11909_small ++ row11909_block_005
theorem row11909_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 11909 row11909_large_left_left :=
  Budget.RowUpperBound.branch row11909_small_covers row11909_block_005_covers

def row11909_large_left_right_right : List Nat := row11909_block_007 ++ row11909_block_008
theorem row11909_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 11909 row11909_large_left_right_right :=
  Budget.RowUpperBound.branch row11909_block_007_covers row11909_block_008_covers

def row11909_large_left_right : List Nat := row11909_block_006 ++ row11909_large_left_right_right
theorem row11909_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 11909 row11909_large_left_right :=
  Budget.RowUpperBound.branch row11909_block_006_covers row11909_large_left_right_right_covers

def row11909_large_left : List Nat := row11909_large_left_left ++ row11909_large_left_right
theorem row11909_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 11909 row11909_large_left :=
  Budget.RowUpperBound.branch row11909_large_left_left_covers row11909_large_left_right_covers

def row11909_large_right_left : List Nat := row11909_block_009 ++ row11909_block_010
theorem row11909_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 11909 row11909_large_right_left :=
  Budget.RowUpperBound.branch row11909_block_009_covers row11909_block_010_covers

def row11909_large_right_right_right : List Nat := row11909_block_012 ++ row11909_block_013
theorem row11909_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 11909 row11909_large_right_right_right :=
  Budget.RowUpperBound.branch row11909_block_012_covers row11909_block_013_covers

def row11909_large_right_right : List Nat := row11909_block_011 ++ row11909_large_right_right_right
theorem row11909_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 11909 row11909_large_right_right :=
  Budget.RowUpperBound.branch row11909_block_011_covers row11909_large_right_right_right_covers

def row11909_large_right : List Nat := row11909_large_right_left ++ row11909_large_right_right
theorem row11909_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 11909 row11909_large_right :=
  Budget.RowUpperBound.branch row11909_large_right_left_covers row11909_large_right_right_covers

def row11909_large : List Nat := row11909_large_left ++ row11909_large_right
theorem row11909_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 11909 row11909_large :=
  Budget.RowUpperBound.branch row11909_large_left_covers row11909_large_right_covers

def row11909 : List Nat := row11909_large
theorem row11909_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 11909 row11909 := row11909_large_covers

def row11923_small : List Nat := []
theorem row11923_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 11923 row11923_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row11923_block_005 : List Nat := []
theorem row11923_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 11923 row11923_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row11923_block_006 : List Nat := [12007, 12211]
def plan11923_block_006 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 19), .first, (.secondFactor 7), .first, .first, .possible, .first, (.firstFactor 7), .first, (.firstFactor 13), (.firstFactor 31), .first, (.firstFactor 19), (.secondFactor 7), .first, .first, (.secondFactor 13), .first, .first, .first, .first, (.firstFactor 23), .first, (.firstFactor 7), .first, .first, .possible, .first, .first, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 13), .first, (.firstFactor 7), (.firstFactor 19), .first, .first, (.secondFactor 13), .first, (.firstFactor 7), .first, (.secondFactor 37), (.firstFactor 17), .first, (.firstFactor 29), .first]
set_option maxRecDepth 2048 in
theorem row11923_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 11923 row11923_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan11923_block_006
  · decide
  · rfl

def row11923_block_007 : List Nat := [12547, 12907, 13729]
def plan11923_block_007 : List Budget.Choice := [(.firstFactor 31), (.firstFactor 13), .first, (.firstFactor 11), (.firstFactor 7), .first, .first, .second, .first, .first, .first, (.firstFactor 13), (.firstFactor 11), .first, .first, (.firstFactor 7), .possible, (.secondFactor 31), .first, (.secondFactor 13), (.firstFactor 7), (.firstFactor 13), (.secondFactor 7), .first, (.secondFactor 17), (.secondFactor 19), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 17), .first, .first, .first, (.secondFactor 31), .first, (.firstFactor 19), .first, (.firstFactor 11), .first, (.firstFactor 17), .first, .first, (.firstFactor 13), .second, .first, (.secondFactor 7), (.secondFactor 13), .first, .first, .possible, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.secondFactor 7), (.secondFactor 29), (.firstFactor 19), .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), (.firstFactor 11), .first, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 29), (.firstFactor 13), (.firstFactor 7), (.firstFactor 11), .first, (.firstFactor 19), .first, .first, (.firstFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 23), (.firstFactor 17), (.secondFactor 19), .first, (.firstFactor 11), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 23), .first, (.firstFactor 37), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 29), (.firstFactor 7), .first, (.firstFactor 19), .first, (.firstFactor 13), .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 37), .first, (.firstFactor 31), .possible, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 37), .first, .first, .first, (.secondFactor 23), .first, .first, (.secondFactor 19), .first, .first, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 17), .first, (.secondFactor 37), (.secondFactor 29), .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 23), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row11923_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 11923 row11923_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan11923_block_007
  · decide
  · rfl

def row11923_block_008 : List Nat := []
theorem row11923_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 11923 row11923_block_008 :=
  Budget.empty_row_after _ _ _ 14466 (fun q member => (Bounds.block_008 q member).1) (by decide)

def row11923_block_009 : List Nat := []
theorem row11923_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 11923 row11923_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row11923_block_010 : List Nat := []
theorem row11923_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 11923 row11923_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row11923_block_011 : List Nat := []
theorem row11923_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 11923 row11923_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row11923_block_012 : List Nat := []
theorem row11923_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 11923 row11923_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row11923_block_013 : List Nat := []
theorem row11923_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 11923 row11923_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row11923_large_left_left : List Nat := row11923_small ++ row11923_block_005
theorem row11923_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 11923 row11923_large_left_left :=
  Budget.RowUpperBound.branch row11923_small_covers row11923_block_005_covers

def row11923_large_left_right_right : List Nat := row11923_block_007 ++ row11923_block_008
theorem row11923_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 11923 row11923_large_left_right_right :=
  Budget.RowUpperBound.branch row11923_block_007_covers row11923_block_008_covers

def row11923_large_left_right : List Nat := row11923_block_006 ++ row11923_large_left_right_right
theorem row11923_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 11923 row11923_large_left_right :=
  Budget.RowUpperBound.branch row11923_block_006_covers row11923_large_left_right_right_covers

def row11923_large_left : List Nat := row11923_large_left_left ++ row11923_large_left_right
theorem row11923_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 11923 row11923_large_left :=
  Budget.RowUpperBound.branch row11923_large_left_left_covers row11923_large_left_right_covers

def row11923_large_right_left : List Nat := row11923_block_009 ++ row11923_block_010
theorem row11923_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 11923 row11923_large_right_left :=
  Budget.RowUpperBound.branch row11923_block_009_covers row11923_block_010_covers

def row11923_large_right_right_right : List Nat := row11923_block_012 ++ row11923_block_013
theorem row11923_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 11923 row11923_large_right_right_right :=
  Budget.RowUpperBound.branch row11923_block_012_covers row11923_block_013_covers

def row11923_large_right_right : List Nat := row11923_block_011 ++ row11923_large_right_right_right
theorem row11923_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 11923 row11923_large_right_right :=
  Budget.RowUpperBound.branch row11923_block_011_covers row11923_large_right_right_right_covers

def row11923_large_right : List Nat := row11923_large_right_left ++ row11923_large_right_right
theorem row11923_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 11923 row11923_large_right :=
  Budget.RowUpperBound.branch row11923_large_right_left_covers row11923_large_right_right_covers

def row11923_large : List Nat := row11923_large_left ++ row11923_large_right
theorem row11923_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 11923 row11923_large :=
  Budget.RowUpperBound.branch row11923_large_left_covers row11923_large_right_covers

def row11923 : List Nat := row11923_large
theorem row11923_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 11923 row11923 := row11923_large_covers

def row11927_small : List Nat := []
theorem row11927_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 11927 row11927_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row11927_block_005 : List Nat := []
theorem row11927_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 11927 row11927_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row11927_block_006 : List Nat := [11981]
def plan11927_block_006 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, (.firstFactor 7), (.secondFactor 13), .first, .first, .first, (.secondFactor 23), .first, .possible, (.secondFactor 7), .first, (.firstFactor 31), .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 7), (.secondFactor 23), .first, (.secondFactor 7), .first, (.firstFactor 7), (.secondFactor 19), .first, (.secondFactor 31), .first, (.firstFactor 17), (.firstFactor 29), .first, (.firstFactor 7), (.secondFactor 7), .first, (.firstFactor 19), .first, (.secondFactor 19), (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.firstFactor 11), (.secondFactor 13), .first, (.secondFactor 31), .first, (.firstFactor 29), .first, .first, (.firstFactor 17), .first, (.secondFactor 37)]
set_option maxRecDepth 2048 in
theorem row11927_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 11927 row11927_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan11927_block_006
  · decide
  · rfl

def row11927_block_007 : List Nat := [12923, 12959, 13103]
def plan11927_block_007 : List Budget.Choice := [.first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 17), .first, .first, (.secondFactor 29), .first, .first, .first, .first, (.firstFactor 31), .first, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 13), (.firstFactor 7), (.firstFactor 11), (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .second, .first, (.firstFactor 17), (.firstFactor 19), .first, .first, .first, .first, .first, .second, (.firstFactor 7), .first, (.firstFactor 13), (.firstFactor 11), .first, .possible, (.firstFactor 7), (.secondFactor 7), .possible, .first, .first, .first, (.firstFactor 7), (.firstFactor 23), .first, .first, .first, .first, (.secondFactor 7), (.secondFactor 17), (.firstFactor 11), .first, .first, .first, .possible, (.firstFactor 7), (.secondFactor 7), (.firstFactor 31), .first, (.firstFactor 7), .first, (.firstFactor 19), .first, .first, .first, (.secondFactor 13), (.firstFactor 17), .first, .first, (.firstFactor 37), .first, .second, .first, .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 7), .first, .first, .second, .first, (.secondFactor 29), .first, .first, .first, (.firstFactor 17), .first, (.secondFactor 17), (.firstFactor 13), (.firstFactor 37), .second, .first, (.firstFactor 7), (.firstFactor 31), .first, (.firstFactor 17), .first, (.firstFactor 23), .first, (.firstFactor 11), .first, .first, (.firstFactor 7), (.firstFactor 19), .first, .first, (.secondFactor 31), .first, .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 7), (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 7), (.secondFactor 17), .first, (.secondFactor 23), (.firstFactor 7), .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 11), (.secondFactor 17), .first, (.secondFactor 7), .first, .first, (.secondFactor 23), .first, (.firstFactor 7), .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 23), .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 31), .first, (.firstFactor 13), .first, (.secondFactor 7), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row11927_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 11927 row11927_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan11927_block_007
  · decide
  · rfl

def row11927_block_008 : List Nat := []
theorem row11927_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 11927 row11927_block_008 :=
  Budget.empty_row_after _ _ _ 14466 (fun q member => (Bounds.block_008 q member).1) (by decide)

def row11927_block_009 : List Nat := []
theorem row11927_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 11927 row11927_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row11927_block_010 : List Nat := []
theorem row11927_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 11927 row11927_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row11927_block_011 : List Nat := []
theorem row11927_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 11927 row11927_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row11927_block_012 : List Nat := []
theorem row11927_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 11927 row11927_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row11927_block_013 : List Nat := []
theorem row11927_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 11927 row11927_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row11927_large_left_left : List Nat := row11927_small ++ row11927_block_005
theorem row11927_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 11927 row11927_large_left_left :=
  Budget.RowUpperBound.branch row11927_small_covers row11927_block_005_covers

def row11927_large_left_right_right : List Nat := row11927_block_007 ++ row11927_block_008
theorem row11927_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 11927 row11927_large_left_right_right :=
  Budget.RowUpperBound.branch row11927_block_007_covers row11927_block_008_covers

def row11927_large_left_right : List Nat := row11927_block_006 ++ row11927_large_left_right_right
theorem row11927_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 11927 row11927_large_left_right :=
  Budget.RowUpperBound.branch row11927_block_006_covers row11927_large_left_right_right_covers

def row11927_large_left : List Nat := row11927_large_left_left ++ row11927_large_left_right
theorem row11927_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 11927 row11927_large_left :=
  Budget.RowUpperBound.branch row11927_large_left_left_covers row11927_large_left_right_covers

def row11927_large_right_left : List Nat := row11927_block_009 ++ row11927_block_010
theorem row11927_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 11927 row11927_large_right_left :=
  Budget.RowUpperBound.branch row11927_block_009_covers row11927_block_010_covers

def row11927_large_right_right_right : List Nat := row11927_block_012 ++ row11927_block_013
theorem row11927_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 11927 row11927_large_right_right_right :=
  Budget.RowUpperBound.branch row11927_block_012_covers row11927_block_013_covers

def row11927_large_right_right : List Nat := row11927_block_011 ++ row11927_large_right_right_right
theorem row11927_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 11927 row11927_large_right_right :=
  Budget.RowUpperBound.branch row11927_block_011_covers row11927_large_right_right_right_covers

def row11927_large_right : List Nat := row11927_large_right_left ++ row11927_large_right_right
theorem row11927_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 11927 row11927_large_right :=
  Budget.RowUpperBound.branch row11927_large_right_left_covers row11927_large_right_right_covers

def row11927_large : List Nat := row11927_large_left ++ row11927_large_right
theorem row11927_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 11927 row11927_large :=
  Budget.RowUpperBound.branch row11927_large_left_covers row11927_large_right_covers

def row11927 : List Nat := row11927_large
theorem row11927_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 11927 row11927 := row11927_large_covers

def row11933_small : List Nat := []
theorem row11933_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 11933 row11933_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row11933_block_005 : List Nat := []
theorem row11933_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 11933 row11933_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row11933_block_006 : List Nat := [12251, 12401]
def plan11933_block_006 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, (.secondFactor 29), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 29), (.firstFactor 7), .first, (.secondFactor 7), .first, .second, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 7), (.firstFactor 19), (.secondFactor 23), .second, .first, (.firstFactor 17), .first, (.firstFactor 7), .second, .first, .first, (.firstFactor 7), .first, .possible, .first, (.firstFactor 11), (.secondFactor 13), .first, (.firstFactor 7), .first, .first, (.firstFactor 7), (.firstFactor 11), .first, (.firstFactor 19), .first, .first, .first, .first, .possible, .first, (.secondFactor 17)]
set_option maxRecDepth 2048 in
theorem row11933_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 11933 row11933_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan11933_block_006
  · decide
  · rfl

def row11933_block_007 : List Nat := []
def plan11933_block_007 : List Budget.Choice := [.first, .first, .first, .first, .first, (.firstFactor 23), .first, .first, (.firstFactor 7), (.firstFactor 13), (.firstFactor 29), .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 17), .first, .first, .first, .first, (.firstFactor 23), .first, .first, .first, (.firstFactor 37), .first, (.firstFactor 13), (.firstFactor 7), (.firstFactor 17), (.firstFactor 19), .first, .first, .second, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 13), (.secondFactor 17), .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 7), (.firstFactor 19), .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 19), .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 13), (.secondFactor 13), .first, .first, .first, (.secondFactor 7), (.secondFactor 23), (.firstFactor 7), (.secondFactor 13), .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .second, .first, (.secondFactor 7), (.secondFactor 37), .first, (.firstFactor 19), .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .second, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 7), (.secondFactor 37), .first, .first, (.firstFactor 17), (.firstFactor 7), .first, (.secondFactor 7), .first, (.secondFactor 19), .first, (.firstFactor 23), .first, .first, .second, (.secondFactor 31), .first, .first, (.firstFactor 11), .first, (.secondFactor 29), .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 7), .first, (.firstFactor 29), .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 23), .first, (.firstFactor 19), .first, (.secondFactor 17), (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.firstFactor 13), .first, (.firstFactor 31), (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 17), .first, (.secondFactor 19), .first, .first, .second, .second, (.firstFactor 13), .first, .first, .first, (.firstFactor 7), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row11933_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 11933 row11933_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan11933_block_007
  · decide
  · rfl

def row11933_block_008 : List Nat := []
theorem row11933_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 11933 row11933_block_008 :=
  Budget.empty_row_after _ _ _ 14466 (fun q member => (Bounds.block_008 q member).1) (by decide)

def row11933_block_009 : List Nat := []
theorem row11933_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 11933 row11933_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row11933_block_010 : List Nat := []
theorem row11933_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 11933 row11933_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row11933_block_011 : List Nat := []
theorem row11933_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 11933 row11933_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row11933_block_012 : List Nat := []
theorem row11933_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 11933 row11933_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row11933_block_013 : List Nat := []
theorem row11933_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 11933 row11933_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row11933_large_left_left : List Nat := row11933_small ++ row11933_block_005
theorem row11933_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 11933 row11933_large_left_left :=
  Budget.RowUpperBound.branch row11933_small_covers row11933_block_005_covers

def row11933_large_left_right_right : List Nat := row11933_block_007 ++ row11933_block_008
theorem row11933_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 11933 row11933_large_left_right_right :=
  Budget.RowUpperBound.branch row11933_block_007_covers row11933_block_008_covers

def row11933_large_left_right : List Nat := row11933_block_006 ++ row11933_large_left_right_right
theorem row11933_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 11933 row11933_large_left_right :=
  Budget.RowUpperBound.branch row11933_block_006_covers row11933_large_left_right_right_covers

def row11933_large_left : List Nat := row11933_large_left_left ++ row11933_large_left_right
theorem row11933_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 11933 row11933_large_left :=
  Budget.RowUpperBound.branch row11933_large_left_left_covers row11933_large_left_right_covers

def row11933_large_right_left : List Nat := row11933_block_009 ++ row11933_block_010
theorem row11933_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 11933 row11933_large_right_left :=
  Budget.RowUpperBound.branch row11933_block_009_covers row11933_block_010_covers

def row11933_large_right_right_right : List Nat := row11933_block_012 ++ row11933_block_013
theorem row11933_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 11933 row11933_large_right_right_right :=
  Budget.RowUpperBound.branch row11933_block_012_covers row11933_block_013_covers

def row11933_large_right_right : List Nat := row11933_block_011 ++ row11933_large_right_right_right
theorem row11933_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 11933 row11933_large_right_right :=
  Budget.RowUpperBound.branch row11933_block_011_covers row11933_large_right_right_right_covers

def row11933_large_right : List Nat := row11933_large_right_left ++ row11933_large_right_right
theorem row11933_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 11933 row11933_large_right :=
  Budget.RowUpperBound.branch row11933_large_right_left_covers row11933_large_right_right_covers

def row11933_large : List Nat := row11933_large_left ++ row11933_large_right
theorem row11933_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 11933 row11933_large :=
  Budget.RowUpperBound.branch row11933_large_left_covers row11933_large_right_covers

def row11933 : List Nat := row11933_large
theorem row11933_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 11933 row11933 := row11933_large_covers

end ProjectEulerSolutions.PrimePairLarge
