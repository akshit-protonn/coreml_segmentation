// Copyright 2019 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import UIKit
import CoreML

class ImageSegmentator {
        
    /// Dedicated DispatchQueue for TF Lite operations.
    private let tfLiteQueue: DispatchQueue

    private var deeplabModel: DeeplabV3

    /// TF Lite Model's input and output shapes.
    private let batchSize: Int
    private let inputImageDim: Int
    private let inputPixelSize: Int
    
    
    /// Label list contains name of all classes the model can regconize.
    private let labelList: [String]
    
    // MARK: - Initialization
    
    /// Load label list from file.
    private static func loadLabelList() -> [String]? {
        guard
            let labelListPath = Bundle.main.path(
                forResource: Constants.labelsFileName,
                ofType: Constants.labelsFileExtension
            )
        else {
            return nil
        }
        
        // Parse label list file as JSON.
        do {
            let data = try Data(contentsOf: URL(fileURLWithPath: labelListPath), options: .mappedIfSafe)
            let jsonResult = try JSONSerialization.jsonObject(with: data, options: .mutableLeaves)
            if let labelList = jsonResult as? [String] { return labelList } else { return nil }
        } catch {
            print("Error parsing label list file as JSON.")
            return nil
        }
    }
    
    /// Create a new Image Segmentator instance.
    static func newInstance(completion: @escaping ((Result<ImageSegmentator>) -> Void)) {
        // Create a dispatch queue to ensure all operations on the Intepreter will run serially.
        let tfLiteQueue = DispatchQueue(label: "org.tensorflow.examples.lite.image_segmentation")
        
        // Run initialization in background thread to avoid UI freeze.
        tfLiteQueue.async {
            
            // Construct the path to the label list file.
            guard let labelList = loadLabelList() else {
                print(
                    "Failed to load the label list file with name: "
                        + "\(Constants.labelsFileName).\(Constants.labelsFileExtension)"
                )
                DispatchQueue.main.async {
                    completion(
                        .error(
                            InitializationError.invalidLabelList(
                                "\(Constants.labelsFileName).\(Constants.labelsFileExtension)"
                            )))
                }
                return
            }
                        
            do {
                // Create an ImageSegmentator instance and return.
                let segmentator = ImageSegmentator(
                    tfLiteQueue: tfLiteQueue,
                    labelList: labelList
                )
                DispatchQueue.main.async {
                    completion(.success(segmentator))
                }
            } catch let error {
                print("Failed to create the deeplab model with error: \(error.localizedDescription)")
                DispatchQueue.main.async {
                    completion(.error(InitializationError.internalError(error)))
                }
                return
            }
        }
    }
    
    /// Initialize Image Segmentator instance.
    fileprivate init(
        tfLiteQueue: DispatchQueue,
        labelList: [String]
    ) {
        self.deeplabModel = DeeplabV3()
        // todo: take from model
        let inputShape = deeplabModel.model.modelDescription.inputDescriptionsByName["ImageTensor"]!.multiArrayConstraint?.shape as [Int]
        // Read input shape from model.
        self.batchSize = inputShape[0]
        self.inputImageDim = inputShape[1]
        self.inputPixelSize = inputShape[3]
                
        // Store label list
        self.labelList = labelList
        
        // Store the dedicated DispatchQueue for TFLite.
        self.tfLiteQueue = tfLiteQueue
    }
    
    static func parseSegOutput(mArr: MLMultiArray) -> [Bool]{
        var isForeground = Array(repeating: false, count: mArr.count)
        
        let ptr = UnsafeMutablePointer<Float32>( OpaquePointer(mArr.dataPointer))
        for i in 0..<mArr.count{
            isForeground[i] = (Int(Float32(ptr[i]))==15)
        }
        return isForeground
    }
    
    // MARK: - Image Segmentation
    
    /// Run segmentation on a given image.
    /// - Parameter image: the target image.
    /// - Parameter completion: the callback to receive segmentation result.
    func runSegmentation(
        _ image: UIImage, completion: @escaping ((Result<SegmentationResult>) -> Void)
    ) {
        tfLiteQueue.async {
            var startTime: Date = Date()
            var preprocessingTime: TimeInterval = 0
            var inferenceTime: TimeInterval = 0
            var postprocessingTime: TimeInterval = 0
            var visualizationTime: TimeInterval = 0
            let modelOutput: DeeplabV3Output
            do {
                // Preprocessing: Resize the input UIImage to match with TF Lite model input shape.
                guard
                    let scaledImage = image.scaledData(
                        with: CGSize(width: self.inputImageDim, height: self.inputImageDim),
                        byteCount: self.inputImageDim * self.inputImageDim * 3 * 1,
                        isQuantized: true
                    )
                else {
                    DispatchQueue.main.async {
                        completion(.error(SegmentationError.invalidImage))
                    }
                    print("Failed to convert the image buffer to RGB data.")
                    return
                }
                
                let modelInput = scaledImage.toArray(type: UInt8.self).map {Float32($0)}
                // https://github.com/huggingface/swift-coreml-transformers/blob/master/Sources/MLMultiArray%2BUtils.swift
                let imageTensor = try! MLMultiArray(
                    shape: [1,self.inputImageDim, self.inputImageDim,3] as [NSNumber],
                    dataType: .float32)
                let ptr = UnsafeMutablePointer<Float32>(OpaquePointer(imageTensor.dataPointer))
                for (i, el) in modelInput.enumerated(){
                    ptr[i] = el
                }
                
                // Calculate preprocessing time.
                var now = Date()
                preprocessingTime = now.timeIntervalSince(startTime)
                // inference time. start
                startTime = Date()
                
                modelOutput = try self.deeplabModel.prediction(ImageTensor: imageTensor)

                // inference time. end
                now = Date()
                inferenceTime = now.timeIntervalSince(startTime)
                startTime = Date()
            } catch let error {
                print("Failed to invoke the deeplab mlmodel with error: \(error.localizedDescription)")
                DispatchQueue.main.async {
                    completion(.error(SegmentationError.internalError(error)))
                }
                return
            }
            
            let parsedOutput = self.parseBinaryClassOutputTensor(predTensor: modelOutput.SemanticPredictions)

            // Calculate postprocessing time.
            // Note: You may find postprocessing very slow if you run the sample app with Debug build.
            // You will see significant speed up if you rerun using Release build, or change
            // Optimization Level in the project's Build Settings to the same value with Release build.
            var now = Date()
            postprocessingTime = now.timeIntervalSince(startTime)
            startTime = Date()
            
            // Visualize result into images.
            guard
                let resultImage = ImageSegmentator.imageFromSRGBColorArray(
                    pixels: parsedOutput.segmentationImagePixels,
                    width: self.inputImageDim,
                    height: self.inputImageDim
                ),
                let overlayImage = image.overlayWithImage(image: resultImage, alpha: 0.5)
            else {
                print("Failed to visualize segmentation result.")
                DispatchQueue.main.async {
                    completion(.error(SegmentationError.resultVisualizationError))
                }
                return
            }
            
            // Construct a dictionary of classes found in the image and each class's color used in
            // visualization.
            let colorLegend = self.classListToColorLegend(classList: parsedOutput.classList)
            
            // Calculate visualization time.
            now = Date()
            visualizationTime = now.timeIntervalSince(startTime)
            
            // Create a representative object that contains the segmentation result.
            let result = SegmentationResult(
                array: parsedOutput.segmentationMap,
                resultImage: resultImage,
                overlayImage: overlayImage,
                preprocessingTime: preprocessingTime,
                inferenceTime: inferenceTime,
                postProcessingTime: postprocessingTime,
                visualizationTime: visualizationTime,
                colorLegend: colorLegend
            )
            
            // Return the segmentation result.
            DispatchQueue.main.async {
                completion(.success(result))
            }
        }
    }
        
    private func parseBinaryClassOutputTensor(predTensor: MLMultiArray)
    -> (segmentationMap: [[Int]], segmentationImagePixels: [UInt32], classList: Set<Int>)
    {
        // Initialize the varibles to store postprocessing result.
        var segmentationMap = [[Int]](
            repeating: [Int](repeating: 0, count: self.inputImageDim),
            count: self.inputImageDim
        )
        var segmentationImagePixels = [UInt32](
            repeating: 0, count: self.inputImageDim * self.inputImageDim)
        let classList: Set<Int> = [0,1]
        
        let pixClass = ImageSegmentator.parseSegOutput(mArr: predTensor)

        // Looping through the output array
        for x in 0..<self.inputImageDim {
            for y in 0..<self.inputImageDim {
                let isForeground = pixClass[x * self.inputImageDim + y]
                segmentationMap[x][y] = isForeground ? 1: 0
                
                // Lookup the color legend for the class.
                let legendColor = Constants.legendColorList[segmentationMap[x][y]]
                segmentationImagePixels[x * self.inputImageDim + y] = legendColor
            }
        }
        return (segmentationMap, segmentationImagePixels, classList)
    }
    
    
    // MARK: - Utils
    
    /// Construct an UIImage from a list of sRGB pixels.
    private static func imageFromSRGBColorArray(pixels: [UInt32], width: Int, height: Int) -> UIImage?
    {
        guard width > 0 && height > 0 else { return nil }
        guard pixels.count == width * height else { return nil }
        
        // Make a mutable copy
        var data = pixels
        
        // Convert array of pixels to a CGImage instance.
        let cgImage = data.withUnsafeMutableBytes { (ptr) -> CGImage in
            let ctx = CGContext(
                data: ptr.baseAddress,
                width: width,
                height: height,
                bitsPerComponent: 8,
                bytesPerRow: MemoryLayout<UInt32>.size * width,
                space: CGColorSpace(name: CGColorSpace.sRGB)!,
                bitmapInfo: CGBitmapInfo.byteOrder32Little.rawValue
                    + CGImageAlphaInfo.premultipliedFirst.rawValue
            )!
            return ctx.makeImage()!
        }
        
        // Convert the CGImage instance to an UIImage instance.
        return UIImage(cgImage: cgImage)
    }
        
    /// Look up the colors used to visualize the classes found in the image.
    private func classListToColorLegend(classList: Set<Int>) -> [String: UIColor] {
        var colorLegend: [String: UIColor] = [:]
        let sortedClassIndexList = classList.sorted()
        sortedClassIndexList.forEach { classIndex in
            // Look up the color legend for the class.
            // Using modulo to reuse colors on segmentation model with large number of classes.
            let color = Constants.legendColorList[classIndex % Constants.legendColorList.count]
            
            // Convert the color from sRGB UInt32 representation to UIColor.
            let a = CGFloat((color & 0xFF00_0000) >> 24) / 255.0
            let r = CGFloat((color & 0x00FF_0000) >> 16) / 255.0
            let g = CGFloat((color & 0x0000_FF00) >> 8) / 255.0
            let b = CGFloat(color & 0x0000_00FF) / 255.0
            colorLegend[labelList[classIndex]] = UIColor(red: r, green: g, blue: b, alpha: a)
        }
        return colorLegend
    }
    
}

// MARK: - Types

/// Callback type for image segmentation request.
typealias ImageSegmentationCompletion = (SegmentationResult?, Error?) -> Void

/// Representation of the image segmentation result.
struct SegmentationResult {
    /// Segmentation result as an array. Each value represents the most likely class the pixel
    /// belongs to.
    let array: [[Int]]
    
    /// Visualization of the segmentation result.
    let resultImage: UIImage
    
    /// Overlay the segmentation result on input image.
    let overlayImage: UIImage
    
    /// Processing time.
    let preprocessingTime: TimeInterval
    let inferenceTime: TimeInterval
    let postProcessingTime: TimeInterval
    let visualizationTime: TimeInterval
    
    /// Dictionary of classes found in the image, and the color used to represent the class in
    /// segmentation result visualization.
    let colorLegend: [String: UIColor]
}

/// Convenient enum to return result with a callback
enum Result<T> {
    case success(T)
    case error(Error)
}

/// Define errors that could happen in the initialization of this class
enum InitializationError: Error {
    // Invalid TF Lite model
    case invalidModel(String)
    
    // Invalid label list
    case invalidLabelList(String)
    
    // TF Lite Internal Error when initializing
    case internalError(Error)
}

/// Define errors that could happen in when doing image segmentation
enum SegmentationError: Error {
    // Invalid input image
    case invalidImage
    
    // TF Lite Internal Error when initializing
    case internalError(Error)
    
    // Invalid input image
    case resultVisualizationError
}

// MARK: - Constants
private enum Constants {
    /// Label list that the segmentation model detects.
    static let labelsFileName = "deeplabv3_labels"
    static let labelsFileExtension = "json"
    
    
    /// List of colors to visualize segmentation result.
    static let legendColorList: [UInt32] = [
        0xFF00_A1C2, // Vivid Blue
        0xFFC1_0020, // Vivid Red
    ]
}
