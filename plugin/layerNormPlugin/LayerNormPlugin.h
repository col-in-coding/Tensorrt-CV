#ifndef TRT_LAYERNORM_PLUGIN_H
#define TRT_LAYERNORM_PLUGIN_H

#include <vector>
#include <string>

#include "NvInfer.h"
#include "NvInferPlugin.h"

// +------- Debug wrapper --------------------------------------------------------------------------
#if DEBUG
#define WHERE_AM_I() do {printf("===> [%s]: this=->%p\n",__func__,this);} while(0);
#else
#define WHERE_AM_I()
#endif // DEBUG

// +------- Plguin ---------------------------------------------------------------------------------
namespace
{
static const char* PLUGIN_NAME{"MyLayerNorm"};
static const char* PLUGIN_VERSION{"1"};
} // namespace

namespace nvinfer1
{

// +------- Plugin body ----------------------------------------------------------------------------
class LayerNormPlugin: public IPluginV2DynamicExt
{
private:    
    std::string name_;
    std::string namespace_;

public:
    // used by PluginCreator
    LayerNormPlugin(const std::string& name) : name_(name)
    {
        WHERE_AM_I();
    }
    // used by deserializing engine file
    LayerNormPlugin(const std::string& name, const void* data, size_t length) : name_(name)
    {
        WHERE_AM_I();
    }
    // default constructor is banned
    LayerNormPlugin() = delete;

    ~LayerNormPlugin()
    {
        WHERE_AM_I();
        terminate();
    }

    size_t getSerializationSize() const noexcept override
    {
        WHERE_AM_I();
        return 0;
    }
    
    void serialize(void *buffer) const noexcept override
    {
        WHERE_AM_I();
    }
  
    IPluginV2DynamicExt* clone() const noexcept override
    {
        WHERE_AM_I();
        return new LayerNormPlugin(name_);
    }

    int getNbOutputs() const noexcept override
    {
        WHERE_AM_I();
        return 1;
    }

    DimsExprs getOutputDimensions(int32_t outputIndex, const DimsExprs* inputs, int32_t nbInputs, IExprBuilder& exprBuilder) noexcept override
    {
        WHERE_AM_I();
        return inputs[0];
    }

    // TODO: support INT8
    bool supportsFormatCombination(int32_t pos, const PluginTensorDesc* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept override
    {
        // inOut[pos].format stands for I/O format
        WHERE_AM_I();

        switch(pos)
        {
        case 0:
            // Input tensor
            return (inOut[0].format == TensorFormat::kLINEAR)
                && (inOut[0].type == DataType::kFLOAT || inOut[0].type == DataType::kHALF);
        case 1:
        case 2:
            // Gamma, Beta
            return (inOut[pos].type == inOut[0].type);
        case 3:
            // Output tensor
            return (inOut[3].type == inOut[0].type) && (inOut[3].format == inOut[0].format);
        default:
            return false;
        }
    }
    
    DataType getOutputDataType(int outputIndex, const DataType* inputTypes, int nbInputs) const noexcept override
    {
        WHERE_AM_I();
        // return inputTypes[0];
        return DataType::kHALF;
    }

    void configurePlugin(const DynamicPluginTensorDesc* in, int32_t nbInputs,const DynamicPluginTensorDesc* out, int32_t nbOutputs) noexcept override
    {
        WHERE_AM_I();
    }

    size_t getWorkspaceSize(const PluginTensorDesc* inputs, int32_t nbInputs, const PluginTensorDesc* outputs,int32_t nbOutputs) const noexcept override
    {
        WHERE_AM_I();
        return 0;
    }

    void setPluginNamespace(const char* szNamespace) noexcept override
    {
        WHERE_AM_I();
        namespace_ = szNamespace;
    }
    const char* getPluginNamespace() const noexcept override
    {
        WHERE_AM_I();
        return namespace_.c_str();
    }
    const char* getPluginType() const noexcept override
    {
        WHERE_AM_I();
        return PLUGIN_NAME;
    }
    const char* getPluginVersion() const noexcept override
    {
        WHERE_AM_I();
        return PLUGIN_VERSION;
    }
    // allocate memory and copy weights to GPU
    int initialize() noexcept override
    {
        WHERE_AM_I();
        return 0;
    }
    // release memory
    void terminate() noexcept override
    {
        WHERE_AM_I();
        return;
    }

    void destroy() noexcept override
    {
        WHERE_AM_I();
    }
    
    int32_t enqueue(const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;
}; // class LayerNormPlugin

class LayerNormPluginCreator : public IPluginCreator
{
private:
    static PluginFieldCollection fc_;
    static std::vector<PluginField> attr_;
    std::string namespace_;

public:
    LayerNormPluginCreator()
    {
        // Define the attributes in Plugin
        fc_.nbFields = attr_.size();
        fc_.fields = attr_.data();
    }

    ~LayerNormPluginCreator() {}

    IPluginV2* createPlugin(const char* name, const PluginFieldCollection* fc) noexcept override
    {
        WHERE_AM_I();
        // PluginField const* fields = fc->fields;
        // printf("... number of fields: [%d]", fc->nbFields);
        // for (int32_t i = 0; i < fc->nbFields; i++)
        // {
        //     printf("... field name: [%s]", fields[i].name);
        // }
        return new LayerNormPlugin(name);
    }

    IPluginV2* deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept override
    {
        return new LayerNormPlugin(name, serialData, serialLength);
    }

    void setPluginNamespace(const char* szNamespace) noexcept override
    {
        namespace_ = szNamespace;
    }

    const char* getPluginNamespace() const noexcept override
    {
        return namespace_.c_str();
    }

    const char* getPluginName() const noexcept override
    {
        return PLUGIN_NAME;
    }

    const char* getPluginVersion() const noexcept override
    {
        return PLUGIN_VERSION;
    }

    const PluginFieldCollection* getFieldNames() noexcept override
    {
        return &fc_;
    }
}; // class LayerNormPluginCreator

} // namespace nvinfer1

#endif